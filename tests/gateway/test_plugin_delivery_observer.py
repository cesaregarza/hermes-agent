"""Final-delivery and queue conservation contracts for plugin injections."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.plugin_delivery import (
    DeliveryObserver, final_reply_display, is_observed_reply_turn, observed_reply_turn,
)
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _DeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.DISCORD)
        self.transport = AsyncMock(return_value=SendResult(success=True, message_id="reply"))

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return await self.transport(chat_id, content, reply_to, metadata)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _source():
    return SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="dm", user_id="42")


def _event(observed, text="completion"):
    event = MessageEvent(text=text, source=_source(), internal=True, allow_gateway_control=False)
    event._plugin_delivery_observer = DeliveryObserver(observed.append)
    return event


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}
    runner._queued_events = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._gateway_loop = asyncio.get_running_loop()
    runner._is_user_authorized = MagicMock(return_value=True)
    runner.session_store = SimpleNamespace()
    source = _source()
    now = datetime.now()
    entry = SessionEntry(session_key=build_session_key(source), session_id="generation-1",
                         created_at=now, updated_at=now, origin=source, platform=Platform.DISCORD)
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry))
    return runner, entry


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["sent", "streamed", "timeout", "retryable", "format_failure", "partial", "no_id", "crashed", "cancelled"])
async def test_delivery_observation_requires_complete_final_send(case):
    adapter = _DeliveryAdapter()
    observed = []
    event = _event(observed)
    session_key = build_session_key(event.source)
    response = "released result MEDIA:/tmp/local-file"
    if case == "retryable":
        adapter.transport.return_value = SendResult(success=False, error="network error", retryable=True)
    if case == "format_failure":
        adapter.transport.return_value = SendResult(success=False, error="bad format")
    if case == "timeout":
        adapter.transport.return_value = SendResult(success=False, error="read timeout")
    if case == "no_id":
        adapter.transport.return_value = SendResult(success=True)
    if case == "partial":
        adapter.transport.return_value = SendResult(
            success=True, message_id="partial-reply", raw_response={"partial_overflow": True})

    async def handler(_event):
        assert is_observed_reply_turn() is True
        if case == "crashed":
            raise RuntimeError("handler failed")
        if case == "cancelled":
            raise asyncio.CancelledError()
        _event._plugin_delivery_observer.reply_ready = True
        return None if case == "streamed" else response

    adapter.set_message_handler(handler)
    adapter._run_processing_hook = AsyncMock()
    adapter._start_typing_refresh = MagicMock(return_value=None)
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock(return_value=False)
    adapter._finish_session_task = MagicMock()
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    adapter._notify_turn_error = AsyncMock(return_value=None)
    adapter._extract_response_content = AsyncMock(side_effect=AssertionError("no media expansion"))
    adapter._deliver_attachments = AsyncMock()
    if case == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await adapter._process_message_background(event, session_key)
    else:
        await adapter._process_message_background(event, session_key)

    assert observed == [True if case == "sent" else None]
    assert is_observed_reply_turn() is False
    adapter._extract_response_content.assert_not_awaited()
    adapter._record_delivery_obligation.assert_not_awaited()
    adapter._notify_turn_error.assert_not_awaited()
    if case not in {"streamed", "crashed", "cancelled"}:
        assert adapter.transport.await_count == 1
        assert adapter.transport.await_args.args[1] == response
    if case in {"streamed", "crashed", "cancelled"}:
        adapter.transport.assert_not_awaited()
    # Later cancellation/rejection callbacks cannot overwrite a terminal observation.
    event._plugin_delivery_observer(False)
    assert len(observed) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["fifo", "queue_cap", "strict_session", "missing_handler", "offline", "cli", "display", "recovery", "provider_failure", "tool_failure", "incomplete_result", "normalized_empty", "empty_sentinel", "guard_notice", "proxy", "tool_start_evidence", "schedule_cancel"])
async def test_observers_survive_queueing_and_rejections(case, tmp_path):
    adapter = _DeliveryAdapter()
    runner, entry = _runner(adapter)
    observed = []
    event = _event(observed, "/approve always")
    assert event.get_command() is None
    if case in {"fifo", "queue_cap"}:
        first_observed = []
        first = _event(first_observed)
        adapter._pending_messages[entry.session_key] = first
        if case == "queue_cap":
            runner._BUSY_QUEUE_MAX_PENDING = 1
        runner._queue_or_replace_pending_event(entry.session_key, event)
        assert adapter._pending_messages[entry.session_key] is first
        assert first.text == "completion"
        assert first_observed == []
        if case == "queue_cap":
            assert observed == [False]
            assert runner._overflow_queue(entry.session_key) in (None, [])
        else:
            assert runner._overflow_queue(entry.session_key) == [event]
            assert observed == []
            pending, text = await runner._run_agent_drain_pending(
                {"final_response": "first"}, adapter, event.source, entry.session_key)
            assert (pending, text) == (None, None)
            assert adapter._pending_messages[entry.session_key] is first
            assert runner._overflow_queue(entry.session_key) == [event]
            # Once the observed head owns a fresh task, promote overflow for the next
            # base-task handoff without recursively consuming it under the wrong event.
            adapter._pending_messages.pop(entry.session_key)
            with observed_reply_turn(first):
                assert await runner._run_agent_drain_pending(
                    {"final_response": "first"}, adapter, first.source, entry.session_key) == (None, None)
            assert adapter._pending_messages[entry.session_key] is event
            assert runner._overflow_queue(entry.session_key) == []
        return
    if case == "strict_session":
        event.metadata = {"gateway_session_key": entry.session_key,
                          "gateway_session_id": "old-generation", "gateway_session_strict": True}
        assert await runner._hmwa_resolve_session(event, event.source) is None
        assert observed == [False]
        adapter.transport.assert_not_awaited()
        return
    if case == "missing_handler":
        await adapter.handle_message(event)
        assert observed == [False]
        assert adapter._background_tasks == set()
        return
    if case in {"offline", "cli"}:
        manager = PluginManager()
        context = PluginContext(PluginManifest(name="notify", key="notify", source="user"), manager)
        if case == "cli":
            manager._cli_ref = SimpleNamespace(_pending_input=[], _interrupt_queue=[], _agent_running=False)
        else:
            context._gateway_injection_allowed = lambda: True
        assert context.inject_message("completion", session_key=entry.session_key,
                                      on_delivery=observed.append) is False
        assert observed == [False]
        return
    if case == "tool_start_evidence":
        ctx = SimpleNamespace(_plugin_delivery_observer=event._plugin_delivery_observer,
                              _status_adapter=adapter, _run_still_current=lambda: True)
        turn_runner = TurnRunner(runner, ctx)
        assert turn_runner._status_live() is False
        turn_runner.progress_callback("subagent.complete", preview="failure")
        assert event._plugin_delivery_observer.tool_started is False
        # This also works from a thread without inherited ContextVars.
        turn_runner.combined_tool_start_callback("call", "send_message", {})
        assert event._plugin_delivery_observer.tool_started is True
        adapter.transport.assert_not_awaited()
        return
    if case == "proxy":
        runner._get_proxy_url = MagicMock(return_value="http://remote.invalid")
        runner._run_agent_via_proxy = AsyncMock()
        with observed_reply_turn(event):
            result = await runner._run_agent_inner(
                message="completion", context_prompt="", history=[], source=event.source,
                session_id=entry.session_id, session_key=entry.session_key)
            assert await runner._hmwa_deliver_turn_response(
                event, event.source, entry, entry.session_key, 1, result, [], "error notice", None, False) is None
        assert observed == [False]
        runner._run_agent_via_proxy.assert_not_awaited()
        adapter.transport.assert_not_awaited()
        return
    if case == "guard_notice":
        adapter.set_message_handler(AsyncMock(return_value="Gateway is draining"))
        adapter._run_processing_hook = AsyncMock()
        adapter._start_typing_refresh = MagicMock(return_value=None)
        adapter._stop_typing_refresh = AsyncMock()
        adapter._fire_post_delivery_callback = AsyncMock()
        adapter._flush_text_debounce_now = AsyncMock(return_value=False)
        adapter._finish_session_task = MagicMock()
        await adapter._process_message_background(event, entry.session_key)
        assert observed == [False]
        adapter.transport.assert_not_awaited()
        return
    if case in {"provider_failure", "tool_failure", "incomplete_result", "normalized_empty", "empty_sentinel"}:
        event._plugin_delivery_observer.tool_started = case == "tool_failure"
        agent_result = {"failed": case in {"provider_failure", "tool_failure"}, "already_sent": False,
                        "completed": case != "incomplete_result", "has_final_reply": case != "normalized_empty"}
        if case == "empty_sentinel":
            agent_result.pop("has_final_reply")
            agent_result["final_response"] = "(empty)"
        with observed_reply_turn(event):
            result = await runner._hmwa_deliver_turn_response(
                event, event.source, entry, entry.session_key, 1,
                agent_result, [], "provider unavailable", None, False)
        assert result is None
        assert observed == [None if case == "tool_failure" else False]
        adapter.transport.assert_not_awaited()
        return
    if case == "recovery":
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        stored = store.get_or_create_session(event.source)
        token = store.mark_turn_active(stored.session_key, allow_auto_resume=False)
        assert token
        assert SessionEntry.from_dict(stored.to_dict()).auto_resume_blocked is True
        assert store.mark_resume_pending(stored.session_key, "shutdown_timeout") is False
        assert store.recover_interrupted_turns() == 0
        assert store.suspend_recently_active() == 0
        assert stored.resume_pending is False
        # A clean unwind retains the exclusion so the legacy recency heuristic cannot replay.
        token = store.mark_turn_active(stored.session_key, allow_auto_resume=False)
        assert store.clear_turn_active(stored.session_key, token) is True
        assert stored.auto_resume_blocked is True
        reloaded = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        assert reloaded.lookup_by_session_key(stored.session_key).auto_resume_blocked is True
        # Unrelated internal work preserves exclusion; a new real user turn restores recovery.
        store.mark_turn_active(stored.session_key, allow_auto_resume=None)
        assert stored.auto_resume_blocked is True
        store.mark_turn_active(stored.session_key)
        assert stored.auto_resume_blocked is False
        assert store.mark_resume_pending(stored.session_key) is True
        return
    if case == "display":
        config = {"display": {"streaming": True, "platforms": {"discord": {"streaming": True}}}}
        display = runner._RunAgentDisplay(user_config=config, platform_key="discord",
                                         tool_progress_enabled=True, needs_progress_queue=True)
        isolated = final_reply_display(display)
        assert isolated.user_config["display"]["platforms"]["discord"]["streaming"] is False
        assert config["display"]["platforms"]["discord"]["streaming"] is True
        assert display.tool_progress_enabled is True
        assert isolated.tool_progress_enabled is False
        assert isolated.needs_progress_queue is False
        with observed_reply_turn(event):
            assert runner._proxy_stream_consumer(event.source, None, None, lambda: True) is None
            assert is_observed_reply_turn() is True
        assert is_observed_reply_turn() is False
        return
    runner._dispatch_plugin_message_injection = AsyncMock()
    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="completion", plugin_id="notify",
        on_delivery=observed.append) is True
    assert observed == []
    task = next(iter(runner._background_tasks))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert observed == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['observed', 'internal', 'ordinary', 'bare'])
async def test_gateway_active_marker_wires_recovery_policy(kind, tmp_path):
    adapter = _DeliveryAdapter()
    runner, _ = _runner(adapter)
    store = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    event = _event([])
    entry = store.get_or_create_session(event.source)
    # Start blocked to prove unrelated internal events preserve the exclusion.
    store.mark_turn_active(entry.session_key, allow_auto_resume=False)
    if kind != 'observed':
        del event._plugin_delivery_observer
    if kind == 'ordinary':
        event.internal = False
    if kind == 'bare':
        event = SimpleNamespace()
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store, mark_turn_active=AsyncMock(side_effect=store.mark_turn_active))
    assert await runner._mark_durable_active_turn(event, entry.session_key)
    assert event._gateway_active_turn_session_key == entry.session_key
    reloaded = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    assert reloaded.lookup_by_session_key(entry.session_key).auto_resume_blocked is (kind in {'observed', 'internal'})


@pytest.mark.asyncio
@pytest.mark.parametrize('with_tail', [False, True])
async def test_orphaned_completion_gets_own_observed_task(with_tail):
    adapter = _DeliveryAdapter()
    runner, entry = _runner(adapter)
    outcomes = []
    orphan = _event(outcomes)
    orphan.metadata = {'gateway_session_key': entry.session_key,
                       'gateway_session_id': entry.session_id, 'gateway_session_strict': True}
    incoming = MessageEvent(text='ordinary question', source=_source())
    tail = MessageEvent(text='older queued question', source=_source())
    overflow = runner._session_state(entry.session_key).conversation.queued_events
    overflow.extend([orphan, tail] if with_tail else [orphan])
    task_events = []

    async def handler(event):
        task_events.append(event)
        if event is incoming:
            assert not is_observed_reply_turn()
            rescued, source, internal = runner._hm_rescue_orphaned_fifo(
                event, event.source, False, entry.session_key)
            assert rescued is incoming and source is incoming.source and not internal
            assert overflow[0] is orphan
            assert outcomes == []
            assert await runner._run_agent_drain_pending(
                {'final_response': 'answer'}, adapter, source, entry.session_key) == (None, None)
            # Leave the ordinary reply out of this transport assertion.
            return None
        if event is orphan:
            assert is_observed_reply_turn()
            assert await runner._hmwa_resolve_session(event, event.source) is not None
            assert await runner._run_agent_drain_pending(
                {'final_response': 'released result'}, adapter, event.source, entry.session_key) == (None, None)
            return await runner._hmwa_deliver_turn_response(
                event, event.source, entry, entry.session_key, 1,
                {'final_response': 'released result', 'completed': True}, [], 'released result', None, False)
        assert event is tail and not is_observed_reply_turn()
        return None

    adapter.set_message_handler(handler)
    adapter._run_processing_hook = AsyncMock()
    adapter._start_typing_refresh = MagicMock(return_value=None)
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock(return_value=False)
    adapter._record_delivery_obligation = AsyncMock()
    adapter._send_with_retry = AsyncMock(side_effect=AssertionError('observed send must not retry'))
    await adapter._process_message_background(incoming, entry.session_key)
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks))
    assert task_events == ([incoming, orphan, tail] if with_tail else [incoming, orphan])
    assert outcomes == [True]
    assert overflow == [] and entry.session_key not in adapter._pending_messages
    adapter.transport.assert_awaited_once()
    assert adapter.transport.await_args.args[1] == 'released result'
    adapter._record_delivery_obligation.assert_not_awaited()


# Use the existing real conversation-loop fixture: the producer and finalizer must
# generate the error result, rather than inserting flags that bypass the bug.
from tests.run_agent.test_92450_outer_error_retry_bound import (  # noqa: E402, F401
    loop_agent, _make_local_frame_raiser,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_kind', ['local_processing_error', 'error_near_max_iterations'])
@pytest.mark.parametrize('tool_started', [False, True])
async def test_real_error_exit_cannot_confirm_released_reply(loop_agent, monkeypatch, exit_kind, tool_started):
    from unittest.mock import patch

    adapter = _DeliveryAdapter()
    runner, entry = _runner(adapter)
    outcomes = []
    event = _event(outcomes)
    event._plugin_delivery_observer.tool_started = tool_started
    if exit_kind == 'local_processing_error':
        raiser = _make_local_frame_raiser()
        target = '_strip_think_blocks'
        failure = lambda text: raiser(TypeError('invalid model response'))
    else:
        loop_agent.max_iterations = 3
        target, failure = '_build_assistant_message', RuntimeError('invalid response assembly')
    with (patch.object(loop_agent, target, side_effect=failure),
          patch.object(loop_agent, '_persist_session'),
          patch.object(loop_agent, '_save_trajectory'),
          patch.object(loop_agent, '_cleanup_task_resources')):
        produced = loop_agent.run_conversation('Summarize the released result')
    assert produced['turn_exit_reason'].startswith(exit_kind)
    assert produced['final_response'] and produced['completed'] is True

    ctx = SimpleNamespace(source=event.source, session_key=entry.session_key,
                          user_config={}, message=event.text, agent_holder=[loop_agent], tools_holder=[[]])
    turn_runner = TurnRunner(runner, ctx)
    runner._provider_routing = None
    runner._resolve_session_agent_runtime = MagicMock(return_value=('test-model', {}))
    runner._resolve_session_reasoning_config = MagicMock(return_value=None)
    runner._resolve_session_service_tier = MagicMock(return_value=None)
    runner._resolve_turn_agent_config = MagicMock(return_value=None)
    # Stub setup/transport only. Keep run_sync's result shaping and the delivery
    # gate real so removing exit-reason propagation makes this regression fail.
    stubs = {'_combined_ephemeral_prompt': '', '_setup_stream_consumer': (None, None, None, False),
             '_resolve_turn_agent': (loop_agent, False), '_wire_turn_agent_callbacks': None,
             '_load_turn_history': ([], None, []), '_prepare_turn_message': (event.text, None),
             '_run_conversation_with_approval': produced, '_finish_stream_consumer': None,
             '_sync_session_after_run': (False, entry.session_id, 0)}
    for name, value in stubs.items():
        monkeypatch.setattr(turn_runner, name, MagicMock(return_value=value))

    shaped_results = []

    async def handler(current):
        shaped = turn_runner.run_sync()
        shaped_results.append(shaped)
        assert shaped['turn_exit_reason'] == produced['turn_exit_reason']
        return await runner._hmwa_deliver_turn_response(
            current, current.source, entry, entry.session_key, 1, shaped, [],
            shaped['final_response'], None, False)

    adapter.set_message_handler(handler)
    adapter._run_processing_hook = AsyncMock()
    adapter._start_typing_refresh = MagicMock(return_value=None)
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock(return_value=False)
    await adapter._process_message_background(event, entry.session_key)
    assert len(shaped_results) == 1
    assert outcomes == [None if tool_started else False]
    assert event._plugin_delivery_observer.reply_ready is False
    adapter.transport.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', [None, 'text_response(finish_reason=stop)',
                                   'fallback_prior_turn_content', 'interpreter_shutdown',
                                   'partial_stream_recovery', 'empty_response_exhausted',
                                   'guardrail_halt', 'compaction_handoff_not_actionable',
                                   'all_retries_exhausted_no_response', 'unknown_future_exit'])
@pytest.mark.parametrize('tool_started', [False, True])
async def test_observed_exit_reason_requires_known_complete_answer(reason, tool_started):
    adapter = _DeliveryAdapter()
    runner, entry = _runner(adapter)
    outcomes = []
    event = _event(outcomes)
    event._plugin_delivery_observer.tool_started = tool_started
    agent_result = {'completed': True, 'final_response': 'some text', 'turn_exit_reason': reason}
    result = await runner._hmwa_deliver_turn_response(
        event, event.source, entry, entry.session_key, 1, agent_result, [], 'some text', None, False)
    valid = reason in {None, 'text_response(finish_reason=stop)', 'fallback_prior_turn_content'}
    assert result == ('some text' if valid else None)
    assert event._plugin_delivery_observer.reply_ready is valid
    # Ready to send is still not a delivery acknowledgement.
    assert outcomes == ([] if valid else [None if tool_started else False])
    adapter.transport.assert_not_awaited()
