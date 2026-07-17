"""Tests for configurable background process notification modes.

The gateway process watcher pushes status updates to users' chats when
background terminal commands run.  ``display.background_process_notifications``
controls verbosity: off | result | error | all (default).

Contributed by @PeterFile (PR #593), reimplemented on current main.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import _UNATTRIBUTED_SESSION_CONTEXT_METADATA_KEY
from gateway.run import GatewayRunner, _parse_session_key
from gateway.session import build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRegistry:
    """Return pre-canned sessions, then None once exhausted."""

    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return False


def _build_runner(monkeypatch, tmp_path, mode: str) -> GatewayRunner:
    """Create a GatewayRunner with a fake config for the given mode."""
    (tmp_path / "config.yaml").write_text(
        f"display:\n  background_process_notifications: {mode}\n",
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner


def _watcher_dict(session_id="proc_test", thread_id=""):
    d = {
        "session_id": session_id,
        "check_interval": 0,
        "platform": "telegram",
        "chat_id": "123",
    }
    if thread_id:
        d["thread_id"] = thread_id
    return d


# ---------------------------------------------------------------------------
# _load_background_notifications_mode unit tests
# ---------------------------------------------------------------------------

class TestLoadBackgroundNotificationsMode:

    def test_defaults_to_all(self, monkeypatch, tmp_path):
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "all"

    def test_reads_config_yaml(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "error"

    def test_env_var_overrides_config(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: error\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "off")
        assert GatewayRunner._load_background_notifications_mode() == "off"

    def test_false_value_maps_to_off(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: false\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "off"

    def test_invalid_value_defaults_to_all(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  background_process_notifications: banana\n"
        )
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
        assert GatewayRunner._load_background_notifications_mode() == "all"


# ---------------------------------------------------------------------------
# _run_process_watcher integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "sessions", "expected_calls", "expected_fragment"),
    [
        # all mode: running output → sends update
        (
            "all",
            [
                SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None),
                None,  # process disappears → watcher exits
            ],
            1,
            "is still running",
        ),
        # result mode: running output → no update
        (
            "result",
            [
                SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None),
                None,
            ],
            0,
            None,
        ),
        # off mode: exited process → no notification
        (
            "off",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        # result mode: exited → notifies
        (
            "result",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            1,
            "finished with exit code 0",
        ),
        # error mode: exit 0 → no notification
        (
            "error",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        # error mode: exit 1 → notifies
        (
            "error",
            [SimpleNamespace(output_buffer="traceback\n", exited=True, exit_code=1)],
            1,
            "finished with exit code 1",
        ),
        # all mode: exited → notifies
        (
            "all",
            [SimpleNamespace(output_buffer="ok\n", exited=True, exit_code=0)],
            1,
            "finished with exit code 0",
        ),
    ],
)
async def test_run_process_watcher_respects_notification_mode(
    monkeypatch, tmp_path, mode, sessions, expected_calls, expected_fragment
):
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    # Patch asyncio.sleep to avoid real delays
    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, mode)
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    assert adapter.send.await_count == expected_calls, (
        f"mode={mode}: expected {expected_calls} sends, got {adapter.send.await_count}"
    )
    if expected_fragment is not None:
        sent_message = adapter.send.await_args.args[1]
        assert expected_fragment in sent_message


@pytest.mark.asyncio
async def test_thread_id_passed_to_send(monkeypatch, tmp_path):
    """thread_id from watcher dict is forwarded as metadata to adapter.send()."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict(thread_id="42"))

    assert adapter.send.await_count == 1
    _, kwargs = adapter.send.call_args
    assert kwargs["metadata"] == {"thread_id": "42"}


@pytest.mark.asyncio
async def test_no_thread_id_sends_no_metadata(monkeypatch, tmp_path):
    """When thread_id is empty, metadata should be None (general topic)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._run_process_watcher(_watcher_dict())

    assert adapter.send.await_count == 1
    _, kwargs = adapter.send.call_args
    assert kwargs["metadata"] is None


@pytest.mark.asyncio
async def test_inject_watch_notification_routes_from_session_store_origin(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="123",
            user_name="Emiliyan",
            message_id="stale-origin-message",
        )
    )

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:group:-100:42",
    }

    await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert synth_event.source.platform == Platform.TELEGRAM
    assert synth_event.source.chat_id == "-100"
    assert synth_event.source.chat_type == "group"
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "123"
    assert synth_event.source.user_name == "Emiliyan"
    assert synth_event.source.message_id is None
    assert synth_event.message_id is None
    assert synth_event.metadata[_UNATTRIBUTED_SESSION_CONTEXT_METADATA_KEY] is True
    assert runner._source_with_trigger_message_id(synth_event).message_id is None


@pytest.mark.asyncio
async def test_inject_watch_notification_uses_event_sender_not_session_creator(
    monkeypatch, tmp_path
):
    """Shared-session completions retain the watcher who started the process."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="alice",
            user_name="Alice",
            message_id="alice-origin-message",
        )
    )
    evt = {
        "session_id": "proc_bob",
        "session_key": "agent:main:telegram:group:-100:42",
        "user_id": "bob",
        "user_name": "Bob",
        "message_id": "bob-trigger-message",
    }

    await runner._inject_watch_notification("[SYSTEM: Bob's process finished]", evt)

    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.source.chat_id == "-100"
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "bob"
    assert synth_event.source.user_name == "Bob"
    assert synth_event.source.user_id_alt is None
    assert synth_event.source.message_id is None
    assert synth_event.message_id == "bob-trigger-message"
    assert _UNATTRIBUTED_SESSION_CONTEXT_METADATA_KEY not in synth_event.metadata
    assert runner._source_with_trigger_message_id(synth_event).message_id == (
        "bob-trigger-message"
    )


@pytest.mark.asyncio
async def test_agent_notification_carries_message_id_reply_anchor(monkeypatch, tmp_path):
    """notify_on_complete injection carries the triggering message_id so the
    synthetic event can be reply-anchored back into a Telegram DM topic.

    Without an anchor, Telegram private-chat topic sends fall back to the main
    chat (see _thread_kwargs_for_send / telegram_dm_topic_reply_fallback)."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="SMOKE_OK\n", exited=True, exit_code=0, command="sleep 1",
    )]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    watcher = {
        "session_id": "proc_anchor",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123:24296",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "24296",
        "message_id": "555",
        "notify_on_complete": True,
    }
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert synth_event.message_id == "555"
    assert synth_event.source.thread_id == "24296"


@pytest.mark.asyncio
async def test_agent_notification_no_message_id_is_tolerated(monkeypatch, tmp_path):
    """A watcher dict without message_id (CLI spawn, pre-upgrade checkpoint)
    still injects — message_id is simply None."""
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(
        output_buffer="done\n", exited=True, exit_code=0, command="sleep 1",
    )]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    watcher = {
        "session_id": "proc_anchorless",
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123:24296",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": "24296",
        "notify_on_complete": True,
    }
    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.message_id is None


@pytest.mark.asyncio
async def test_inject_watch_notification_carries_message_id_reply_anchor(monkeypatch, tmp_path):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    runner.session_store._entries["agent:main:telegram:dm:123:24296"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            thread_id="24296",
            user_id="1",
            user_name="Fabio",
            message_id="stale-origin-message",
        )
    )

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:dm:123:24296",
        "message_id": "777",
    }

    await runner._inject_watch_notification("[SYSTEM: Background process matched]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.message_id == "777"
    assert synth_event.source.thread_id == "24296"
    assert synth_event.source.message_id is None
    assert runner._source_with_trigger_message_id(synth_event).message_id == "777"


def test_build_process_event_source_falls_back_to_session_key_chat_type(monkeypatch, tmp_path):
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_watch",
        "session_key": "agent:main:telegram:group:-100:42",
        "platform": "telegram",
        "chat_id": "-100",
        "thread_id": "42",
        "user_id": "123",
        "user_name": "Emiliyan",
    }

    source = runner._build_process_event_source(evt)

    assert source is not None
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "42"
    assert source.user_id == "123"
    assert source.user_name == "Emiliyan"


def test_build_process_event_source_derives_named_profile_from_session_key(
    monkeypatch, tmp_path
):
    runner = _build_runner(monkeypatch, tmp_path, "all")

    source = runner._build_process_event_source(
        {
            "session_id": "proc_coder",
            "session_key": "agent:coder:telegram:dm:123",
        }
    )

    assert source is not None
    assert source.platform is Platform.TELEGRAM
    assert source.chat_type == "dm"
    assert source.chat_id == "123"
    assert source.profile == "coder"


def test_build_process_event_source_uses_cached_live_source_before_session_key_parse(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner._cache_session_source(
        "agent:main:telegram:group:-100:42",
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        ),
    )

    source = runner._build_process_event_source(
        {
            "session_id": "proc_watch",
            "session_key": "agent:main:telegram:group:-100:42",
        }
    )

    assert source is not None
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "42"
    assert source.user_id == "proc_owner"
    assert source.user_name == "alice"


def test_build_process_event_source_uses_coherent_store_provenance_snapshot(
    monkeypatch, tmp_path
):
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    session_key = "agent:ops:discord:dm:relay-chat"
    snapshot_calls = []

    def routing_snapshot(key):
        snapshot_calls.append(key)
        return SessionSource(
            platform=Platform.DISCORD,
            chat_id="relay-chat",
            chat_type="dm",
            user_id="relay-owner",
            message_id="stale-trigger",
            profile="ops",
            transport_profile="primary-bot",
            delivered_via_upstream_relay=True,
        )

    monkeypatch.setattr(
        runner.session_store,
        "routing_source_snapshot",
        routing_snapshot,
    )
    runner._cache_session_source(
        session_key,
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="wrong-cache-destination",
        ),
    )

    source = runner._build_process_event_source(
        {
            "session_id": "proc-relay",
            "session_key": session_key,
        }
    )

    assert snapshot_calls == [session_key]
    assert source is not None
    assert source.platform is Platform.DISCORD
    assert source.chat_id == "relay-chat"
    assert source.message_id is None
    assert source.profile == "ops"
    assert source.transport_profile == "primary-bot"
    assert source.delivered_via_upstream_relay is True


@pytest.mark.asyncio
async def test_multiplex_process_fallback_without_transport_provenance_fails_closed(
    monkeypatch,
    tmp_path,
):
    """A runtime namespace cannot identify which bot received the parent turn."""
    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    primary = SimpleNamespace(handle_message=AsyncMock())
    secondary = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._profile_adapters = {
        "coder": {Platform.TELEGRAM: secondary},
    }

    accepted = await runner._inject_watch_notification(
        "[SYSTEM: ambiguous completion]",
        {
            "session_id": "proc-ambiguous",
            "session_key": "agent:coder:telegram:dm:shared-chat",
        },
    )

    assert accepted is None
    primary.handle_message.assert_not_awaited()
    secondary.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_primary_routed_completion_uses_durable_transport_owner(
    monkeypatch,
    tmp_path,
):
    """Runtime coder traffic received by the shared bot must reply via that bot."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    primary = SimpleNamespace(handle_message=AsyncMock())
    secondary = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._profile_adapters = {
        "coder": {Platform.TELEGRAM: secondary},
    }
    monkeypatch.setattr(
        runner.session_store,
        "routing_source_snapshot",
        lambda _key: SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="shared-chat",
            chat_type="dm",
            user_id="operator",
            profile="coder",
            transport_profile="default",
        ),
    )

    accepted = await runner._inject_watch_notification(
        "[SYSTEM: routed completion]",
        {
            "session_id": "proc-shared-primary",
            "session_key": "agent:coder:telegram:dm:shared-chat",
        },
    )

    assert accepted is True
    primary.handle_message.assert_awaited_once()
    secondary.handle_message.assert_not_awaited()
    event = primary.handle_message.await_args.args[0]
    assert event.source.profile == "coder"
    assert event.source.transport_profile == "default"


def test_process_event_preserves_alt_id_for_same_signal_sender(monkeypatch, tmp_path):
    """A watcher overlay must not move a Signal turn out of its UUID lane."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    origin = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="signal-group",
        chat_type="group",
        user_id="+15551234567",
        user_id_alt="signal-user-uuid",
        user_name="Alice",
    )
    session_key = build_session_key(
        origin,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.session_store._entries[session_key] = SimpleNamespace(origin=origin)

    source = runner._build_process_event_source(
        {
            "session_id": "proc_signal",
            "session_key": session_key,
            "user_id": "+15551234567",
            "user_name": "Alice",
        }
    )

    assert source is not None
    assert source.user_id == "+15551234567"
    assert source.user_id_alt == "signal-user-uuid"
    assert build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    ) == session_key


@pytest.mark.asyncio
async def test_inject_watch_notification_ignores_foreground_event_source(monkeypatch, tmp_path):
    """Negative test: watch notification must NOT route to the foreground thread."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]

    # Session store has the process's original thread (thread 42)
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="proc_owner",
            user_name="alice",
        )
    )

    # The evt dict carries the correct session_key — NOT a foreground event
    evt = {
        "session_id": "proc_cross_thread",
        "session_key": "agent:main:telegram:group:-100:42",
    }

    await runner._inject_watch_notification("[SYSTEM: watch match]", evt)

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    # Must route to thread 42 (process origin), NOT some other thread
    assert synth_event.source.thread_id == "42"
    assert synth_event.source.user_id == "proc_owner"


@pytest.mark.asyncio
async def test_inject_watch_notification_uses_relay_transport_for_relay_origin(
    monkeypatch, tmp_path
):
    """Stored relay origins keep their process-shared transport on wakeup."""
    from gateway.session import SessionSource

    runner = _build_runner(monkeypatch, tmp_path, "all")
    relay_adapter = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters = {Platform.RELAY: relay_adapter}
    session_key = "agent:main:discord:dm:relay-chat"
    runner.session_store._entries[session_key] = SimpleNamespace(
        session_id="gateway-relay-session",
        origin=SessionSource(
            platform=Platform.DISCORD,
            chat_id="relay-chat",
            chat_type="dm",
            user_id="relay-owner",
            delivered_via_upstream_relay=True,
        ),
        origin_transport="relay",
        origin_transport_profile=None,
    )

    accepted = await runner._inject_watch_notification(
        "[SYSTEM: relay watch match]",
        {
            "session_id": "proc-relay",
            "session_key": session_key,
        },
    )

    assert accepted is True
    relay_adapter.handle_message.assert_awaited_once()
    event = relay_adapter.handle_message.await_args.args[0]
    assert event.source.platform is Platform.DISCORD
    assert event.source.delivered_via_upstream_relay is True


def test_build_process_event_source_returns_none_for_empty_evt(monkeypatch, tmp_path):
    """Missing session_key and no platform metadata → None (drop notification)."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    source = runner._build_process_event_source({"session_id": "proc_orphan"})
    assert source is None


def test_build_process_event_source_returns_none_for_invalid_platform(monkeypatch, tmp_path):
    """Invalid platform string → None."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_bad",
        "platform": "not_a_real_platform",
        "chat_type": "dm",
        "chat_id": "123",
    }
    source = runner._build_process_event_source(evt)
    assert source is None


def test_build_process_event_source_returns_none_for_short_session_key(monkeypatch, tmp_path):
    """Session key with <5 parts doesn't parse, falls through to empty metadata → None."""
    runner = _build_runner(monkeypatch, tmp_path, "all")

    evt = {
        "session_id": "proc_short",
        "session_key": "agent:main:telegram",  # Too few parts
    }
    source = runner._build_process_event_source(evt)
    assert source is None


# ---------------------------------------------------------------------------
# _parse_session_key helper
# ---------------------------------------------------------------------------

def test_parse_session_key_valid():
    result = _parse_session_key("agent:main:telegram:group:-100")
    assert result == {"platform": "telegram", "chat_type": "group", "chat_id": "-100"}


def test_parse_session_key_named_profile():
    result = _parse_session_key("agent:coder:telegram:dm:123")
    assert result == {
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "profile": "coder",
    }


def test_parse_session_key_with_extra_parts():
    """6th part in a group key may be a user_id, not a thread_id — omit it."""
    result = _parse_session_key("agent:main:discord:group:chan123:thread456")
    assert result == {"platform": "discord", "chat_type": "group", "chat_id": "chan123"}


def test_parse_session_key_with_user_id_part():
    """Group keys with per-user isolation have user_id as 6th part — don't return as thread_id."""
    result = _parse_session_key("agent:main:telegram:group:chat1:user99")
    assert result == {"platform": "telegram", "chat_type": "group", "chat_id": "chat1"}


def test_parse_session_key_dm_with_thread():
    """DM keys use parts[5] as thread_id unambiguously."""
    result = _parse_session_key("agent:main:telegram:dm:chat1:topic42")
    assert result == {"platform": "telegram", "chat_type": "dm", "chat_id": "chat1", "thread_id": "topic42"}


def test_parse_session_key_thread_chat_type():
    """Thread-typed keys use parts[5] as thread_id unambiguously."""
    result = _parse_session_key("agent:main:discord:thread:chan1:thread99")
    assert result == {"platform": "discord", "chat_type": "thread", "chat_id": "chan1", "thread_id": "thread99"}


def test_parse_session_key_too_short():
    assert _parse_session_key("agent:main:telegram") is None
    assert _parse_session_key("") is None


def test_parse_session_key_wrong_prefix():
    assert _parse_session_key("cron:main:telegram:dm:123") is None
    assert _parse_session_key("agent:bad.profile:telegram:dm:123") is None
