"""Event and profile boundaries for external metadata, including recursive queued turns."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
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


def _discord_turn(runner, **source_fields):
    source = SessionSource(platform=Platform.DISCORD, user_id="alice", chat_id="chan", message_id="msg",
                           guild_id="300", **source_fields)
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_id="sid", session_key="key")
    runner._set_session_env(context)
    from gateway.session_context import _SESSION_REDACT_PII
    _SESSION_REDACT_PII.set(False)
    return build_session_context_meta("edge")


def test_server_message_forwards_role_ids_and_a_dm_forwards_none(runner):
    assert _discord_turn(runner, role_ids=("600", "601"))[PREFIX + "role_ids"] == "600,601"
    assert _discord_turn(runner, role_ids=())[PREFIX + "role_ids"] == ""
    assert PREFIX + "role_ids" not in _discord_turn(runner, role_ids=None)


def test_thread_message_forwards_its_parent_channel_only(runner):
    thread = _discord_turn(runner, chat_type="thread", thread_id="thread", parent_chat_id="parent")
    assert thread[PREFIX + "parent_chat_id"] == "parent"
    assert PREFIX + "parent_chat_id" not in _discord_turn(runner, parent_chat_id="parent")


def test_redacted_turn_forwards_neither_field(runner):
    _discord_turn(runner, role_ids=("600",), thread_id="thread", parent_chat_id="parent")
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


def test_role_ids_never_enter_the_stored_origin():
    source = SessionSource(platform=Platform.DISCORD, chat_id="chan", guild_id="300", role_ids=("600",))
    stored = source.to_dict()
    assert "role_ids" not in stored and "600" not in repr(stored)
    assert SessionSource.from_dict(stored).role_ids is None


def test_neither_field_enters_a_subprocess_environment(runner):
    from tools.environments.local import build_subprocess_env

    _discord_turn(runner, role_ids=("600000000000000600",), chat_type="thread", thread_id="thread",
                  parent_chat_id="700000000000000700")
    env = build_subprocess_env()
    assert not any(name.endswith(("ROLE_IDS", "PARENT_CHAT_ID")) for name in env)
    assert not any(value in ("600000000000000600", "700000000000000700") for value in env.values())
