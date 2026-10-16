"""Event and profile boundaries for external metadata, including recursive queued turns."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, PlatformConfig
from gateway.run import GatewayRunner
from gateway.run_session_metadata import bind_session_context_for_turn
from gateway.session import SessionContext, SessionSource
from gateway.session_context import reset_session_vars
from tools import mcp_tool as core
from tools.mcp_tool_session_metadata import build_session_context_meta

PREFIX = "com.nousresearch.hermes/"


@pytest.fixture
def runner(monkeypatch):
    import gateway.run as gateway_run
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(gateway_run, "_hermes_home", get_hermes_home())
    monkeypatch.setattr(core, "_session_context_forwarding_servers", {"edge"})
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    yield runner
    reset_session_vars()


@pytest.mark.parametrize("policy", ["true", "false", "bad", "[", "managed", "missing-profile"])
def test_routed_policy_is_snapshotted_and_fails_closed(runner, tmp_path, monkeypatch, policy):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles, managed_scope

    default = get_hermes_home()
    (default / "config.yaml").write_text("privacy:\n  redact_pii: false\n", encoding="utf-8")
    routed = tmp_path / "routed"
    routed.mkdir()
    (routed / "config.yaml").write_text(f"privacy:\n  redact_pii: {policy if policy != 'managed' else 'false'}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: routed)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: policy != "missing-profile")
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    if policy == "managed":
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text("privacy:\n  redact_pii: true\n", encoding="utf-8")
        monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: managed)
    runner.config.multiplex_profiles = True
    source = SessionSource(platform=Platform.TELEGRAM, user_id="alice", chat_id="chat", profile="secondary", message_id="msg")
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_id="sid", session_key="key")
    _, snapshot = bind_session_context_for_turn(runner, context)
    (routed / "config.yaml").write_text("privacy:\n  redact_pii: false\n", encoding="utf-8")
    meta = build_session_context_meta("edge")
    if policy in {"bad", "[", "missing-profile"}:
        assert snapshot is None and meta is None
    else:
        assert snapshot is (policy in {"true", "managed"})
        assert (meta[PREFIX + "user_id"] != "alice") is snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic", [False, True])
async def test_recursive_queued_turn_rebinds_sender_and_trigger(runner, synthetic):
    first = SessionSource(platform=Platform.TELEGRAM, user_id="alice", chat_id="shared", message_id="stale")
    initial_event = MessageEvent(source=first, text="start", message_type=MessageType.TEXT, message_id="first")
    entry = SimpleNamespace(session_id="sid", session_key="key", created_at=None, updated_at=None)
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))

    class Prepared(Exception):
        pass

    def capture_initial(*args):
        meta = build_session_context_meta("edge")
        assert meta[PREFIX + "message_id"] == "first"
        assert meta[PREFIX + "session_id"] == "sid"
        raise Prepared

    runner._pinned_session_context_prompt = capture_initial
    with pytest.raises(Prepared):
        await runner._hmwa_prepare_turn(initial_event, first, entry, "key", "key", 1)
    first = initial_event.source
    next_source = SessionSource(platform=Platform.TELEGRAM, user_id="alice" if synthetic else "bob", chat_id="shared",
                                message_id="origin" if synthetic else None)
    event = MessageEvent(source=next_source, text="continue", message_type=MessageType.TEXT,
                         message_id=None if synthetic else "second", internal=synthetic)
    turn = SimpleNamespace(source=first, session_id="sid", session_key="key", run_generation=1,
                           _interrupt_depth=0, history=[], _status_thread_metadata={}, context_prompt="pinned")
    runner._adapter_for_source = lambda source: None
    runner._session_key_for_source = lambda source: "key"
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="continue")
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    observed = []

    async def run_followup(**kwargs):
        observed.append(build_session_context_meta("edge"))
        assert kwargs["context_prompt"] == "pinned"
        return {"final_response": "ok", "messages": []}

    runner._run_agent = run_followup
    await runner._run_agent_queued_followup(turn, None, "continue", event, "done", {"messages": []}, None)
    assert observed[0][PREFIX + "user_id"] == next_source.user_id
    assert observed[0][PREFIX + "message_id"] == ("origin" if synthetic else "second")
    assert observed[0][PREFIX + "session_id"] == "sid"



def _discord_turn(runner, event_roles=None, **source_fields):
    """Bind one Discord turn as the gateway does: the source, and the turn's own event."""
    source = SessionSource(platform=Platform.DISCORD, user_id="alice", chat_id="chan", message_id="msg",
                           guild_id="300", **source_fields)
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_id="sid", session_key="key")
    event = MessageEvent(text="hi", source=source, discord_role_ids=event_roles)
    bind_session_context_for_turn(runner, context, event)
    from gateway.session_context import _SESSION_REDACT_PII
    _SESSION_REDACT_PII.set(False)
    return build_session_context_meta("edge")


def test_server_message_forwards_role_ids_and_a_dm_forwards_none(runner):
    assert _discord_turn(runner, ("600", "601"))[PREFIX + "role_ids"] == "600,601"
    assert _discord_turn(runner, ())[PREFIX + "role_ids"] == ""
    assert PREFIX + "role_ids" not in _discord_turn(runner, None)


def test_thread_message_forwards_its_parent_channel_only(runner):
    thread = _discord_turn(runner, chat_type="thread", thread_id="thread", parent_chat_id="parent")
    assert thread[PREFIX + "parent_chat_id"] == "parent"
    assert PREFIX + "parent_chat_id" not in _discord_turn(runner, parent_chat_id="parent")


def test_redacted_turn_forwards_neither_field(runner):
    _discord_turn(runner, ("600",), thread_id="thread", parent_chat_id="parent")
    from gateway.session_context import _SESSION_REDACT_PII
    _SESSION_REDACT_PII.set(True)
    meta = build_session_context_meta("edge")
    assert PREFIX + "role_ids" not in meta and PREFIX + "parent_chat_id" not in meta


def test_role_ids_drop_everyone_and_are_none_in_a_dm():
    from tools.mcp_tool_session_metadata import discord_role_ids

    guild = SimpleNamespace(id=300)
    member = SimpleNamespace(roles=[SimpleNamespace(id=300), SimpleNamespace(id=601), SimpleNamespace(id=600)])
    assert discord_role_ids(member, guild) == ("600", "601")
    assert discord_role_ids(SimpleNamespace(id=1), guild) == ()  # a webhook or plain user
    assert discord_role_ids(member, None) is None


def test_a_bind_that_does_not_set_roles_has_none(runner):
    from gateway.session_context import set_session_vars

    _discord_turn(runner, ("600",), chat_type="thread", thread_id="thread", parent_chat_id="parent")
    # A path that binds with set_session_vars alone (cron, the API server, ACP, the TUI).
    set_session_vars(platform="discord", chat_id="chan", user_id="alice", session_key="key",
                     session_id="sid", message_id="msg", thread_id="thread", redact_pii=False)
    meta = build_session_context_meta("edge")
    assert PREFIX + "role_ids" not in meta and PREFIX + "parent_chat_id" not in meta


def test_neither_field_enters_a_subprocess_environment(runner):
    from tools.environments.local import build_subprocess_env

    _discord_turn(runner, ("600000000000000600",), chat_type="thread", thread_id="thread",
                  parent_chat_id="700000000000000700")
    env = build_subprocess_env()
    assert not any(name.endswith(("ROLE_IDS", "PARENT_CHAT_ID")) for name in env)
    assert not any(value in ("600000000000000600", "700000000000000700") for value in env.values())


KEPT, REVOKED = "600000000000000600", "600000000000000601"


class _DiscordAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("network send is not expected")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def _live_discord_session(runner, tmp_path, adapter):
    """A real store's live Discord session, and a way to bind a turn and read its roles."""
    from gateway.session import SessionStore

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    origin = SessionSource(platform=Platform.DISCORD, chat_id="chan", chat_type="group",
                           user_id="alice", guild_id="300", message_id="m0")
    entry = store.get_or_create_session(origin)
    runner.session_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}
    runner._running, runner._draining = True, False
    runner._is_user_authorized = lambda *args, **kwargs: True
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    seen: list[dict | None] = []

    class Prepared(Exception):
        pass

    def capture(*args):
        seen.append(build_session_context_meta("edge"))
        raise Prepared

    runner._pinned_session_context_prompt = capture

    async def turn(event):
        """The `role_ids` the real binder and builder send for this turn ("absent" if none)."""
        with pytest.raises(Prepared):
            await runner._hmwa_prepare_turn(event, event.source, entry, entry.session_key, entry.session_key, 1)
        return seen.pop().get(PREFIX + "role_ids", "absent")

    def message(message_id, roles, *, media=False):
        source = dataclasses.replace(origin, message_id=message_id)
        if media:
            return MessageEvent(text=f"caption {message_id}", message_type=MessageType.PHOTO, source=source,
                                message_id=message_id, media_urls=[f"{message_id}.jpg"],
                                media_types=["image/jpeg"], discord_role_ids=roles)
        return MessageEvent(text=f"text {message_id}", source=source, message_id=message_id,
                            discord_role_ids=roles)

    return SimpleNamespace(store=store, origin=origin, entry=entry, turn=turn, message=message)


@pytest.mark.asyncio
async def test_no_retained_source_replays_roles_after_a_revocation_or_a_roleless_message(
    runner, tmp_path, monkeypatch,
):
    """Roles come only from each turn's own message (CES-1448 review P1).

    The real store keeps the inbound source as its live origin. After a message with
    roles, one with fewer (a revocation) and one with none, a plugin injection and a
    background notification on the same session, both built from that origin, carry
    no roles, and nothing the store keeps holds any.
    """
    dispatched: list[MessageEvent] = []
    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=dispatched.append))
    live = _live_discord_session(runner, tmp_path, adapter)
    entry = live.entry

    assert await live.turn(live.message("m1", (KEPT, REVOKED))) == f"{KEPT},{REVOKED}"
    assert await live.turn(live.message("m2", (KEPT,))) == KEPT  # one role revoked
    assert await live.turn(live.message("m3", ())) == ""  # a roleless follow-up

    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="plugin says hi", plugin_id="p",
    ) is True
    monkeypatch.setattr(runner, "_build_process_event_source", lambda evt: entry.origin)
    monkeypatch.setattr(runner, "_resolve_injection_adapter", lambda name: adapter)
    assert await runner._inject_watch_notification("job finished", {"session_id": "proc"}) is True
    plugin_event, background_event = dispatched
    assert plugin_event.source.message_id == background_event.source.message_id == "m0"  # the stored origin
    assert await live.turn(plugin_event) == "absent"
    assert await live.turn(background_event) == "absent"

    stored = live.store._entries[entry.session_key].origin
    assert not hasattr(stored, "role_ids") and not hasattr(stored, "discord_role_ids")
    persisted = b"".join(path.read_bytes() for path in (tmp_path / "sessions").rglob("*") if path.is_file())
    assert KEPT.encode() not in persisted and REVOKED.encode() not in persisted


# The later message's roles beside the earlier one's (KEPT, REVOKED), and what the merged turn sends.
MERGED_ROLES = {
    "unchanged": ((KEPT, REVOKED), f"{KEPT},{REVOKED}"),
    "one revoked": ((KEPT,), "absent"),
    "roleless follow-up": ((), "absent"),
    "no testimony": (None, "absent"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MERGED_ROLES)
async def test_busy_media_merged_into_one_turn_keeps_roles_only_if_every_message_agrees(
    runner, tmp_path, case,
):
    adapter = _DiscordAdapter()
    live = _live_discord_session(runner, tmp_path, adapter)
    later_roles, sent = MERGED_ROLES[case]
    key = live.entry.session_key
    adapter._pending_messages[key] = live.message("m1", (KEPT, REVOKED), media=True)

    runner._queue_or_replace_pending_event(key, live.message("m2", later_roles, media=True))
    drained = adapter._pending_messages.pop(key)

    assert drained.media_urls == ["m1.jpg", "m2.jpg"]  # absorbed into one turn
    assert await live.turn(drained) == sent


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MERGED_ROLES)
async def test_busy_text_debounced_into_one_turn_keeps_roles_only_if_every_message_agrees(
    runner, tmp_path, case,
):
    adapter = _DiscordAdapter()
    live = _live_discord_session(runner, tmp_path, adapter)
    later_roles, sent = MERGED_ROLES[case]
    key = live.entry.session_key

    await adapter._queue_text_debounce(key, live.message("m1", (KEPT, REVOKED)))
    await adapter._queue_text_debounce(key, live.message("m2", later_roles))
    assert await adapter._flush_text_debounce_now(key) is True
    drained = adapter._pending_messages.pop(key)

    assert drained.text == "text m1\ntext m2"  # one turn
    assert await live.turn(drained) == sent
