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
