from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import sys

import pytest

from run_agent import AIAgent


def _mock_response(*, usage: dict, content: str = "done"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(**usage),
    )


def _make_agent(session_db, *, platform: str):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=f"{platform}-session",
            platform=platform,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response(
        usage={
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }
    )
    return agent


def test_run_conversation_persists_tokens_for_telegram_sessions():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="telegram")

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    session_db.update_token_counts.assert_called_once()
    assert session_db.update_token_counts.call_args.args[0] == "telegram-session"


def test_run_conversation_persists_tokens_for_cron_sessions():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="cron")

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    session_db.update_token_counts.assert_called_once()
    assert session_db.update_token_counts.call_args.args[0] == "cron-session"


def test_session_search_lazily_opens_db_when_entrypoint_did_not_pass_one(monkeypatch):
    sentinel_db = object()
    captured = {}

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = FakeSessionDB
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)

    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True, "results": []})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(None, platform="acp")
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes", "profile": "work"},
        "task-id",
    ))

    assert result["success"] is True
    assert captured["db"] is sentinel_db
    assert captured["query"] == "Hermes"
    assert captured["profile"] == "work"
    assert captured["profile_name"] is None
    assert captured["allow_cross_profile"] is True
    assert agent._session_db is sentinel_db


@pytest.mark.parametrize(
    ("owner", "profile_name", "gateway_key"),
    [
        ("research", "research", "agent:main:telegram:dm:1"),
        ("default", "coder", "agent:coder:telegram:dm:2"),
    ],
)
def test_gateway_session_search_passes_classified_profile_boundary(
    tmp_path,
    monkeypatch,
    owner,
    profile_name,
    gateway_key,
):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / f"{owner}.db", profile_name=owner)
    db.create_session(
        "telegram-session",
        source="telegram",
        session_key=gateway_key,
        profile_name=profile_name,
    )
    captured = {}
    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True, "results": []})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(db, platform="telegram")
    agent._gateway_session_key = gateway_key
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes", "profile": "untrusted-model-value"},
        "task-id",
    ))

    assert result["success"] is True
    assert captured["profile"] == "untrusted-model-value"
    assert captured["profile_name"] == profile_name
    assert captured["allow_cross_profile"] is False


def test_gateway_session_search_rejects_incoherent_current_row(
    tmp_path,
    monkeypatch,
):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db", profile_name="default")
    db.create_session(
        "telegram-session",
        source="telegram",
        session_key="agent:coder:telegram:dm:2",
        profile_name="default",
    )
    called = False
    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**_kwargs):
        nonlocal called
        called = True
        return json.dumps({"success": True})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(db, platform="telegram")
    agent._gateway_session_key = "agent:coder:telegram:dm:2"
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes"},
        "task-id",
    ))

    assert result["success"] is False
    assert "invalid persisted profile evidence" in result["error"]
    assert called is False


def test_gateway_session_search_uses_routing_key_before_row_is_persisted(
    tmp_path,
    monkeypatch,
):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db", profile_name="default")
    captured = {}
    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(db, platform="telegram")
    agent._gateway_session_key = "agent:coder:telegram:dm:2"
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes"},
        "task-id",
    ))

    assert result["success"] is True
    assert captured["profile_name"] == "coder"
    assert captured["allow_cross_profile"] is False


@pytest.mark.parametrize("memory_key", [None, "webui:user-42"])
def test_api_gateway_session_search_uses_authoritative_profile_not_memory_key(
    tmp_path,
    monkeypatch,
    memory_key,
):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db", profile_name="default")
    db.create_session(
        "api_server-session",
        source="api_server",
        profile_name="coder",
    )
    captured = {}
    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(db, platform="api_server")
    agent._gateway_session_key = memory_key
    agent._gateway_session_search_profile = "coder"
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes", "profile": "model-controlled"},
        "task-id",
    ))

    assert result["success"] is True
    assert captured["profile_name"] == "coder"
    assert captured["allow_cross_profile"] is False


def test_api_gateway_session_search_fails_closed_without_profile_boundary(
    tmp_path,
    monkeypatch,
):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db", profile_name="default")
    called = False
    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**_kwargs):
        nonlocal called
        called = True
        return json.dumps({"success": True})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(db, platform="api_server")
    agent._gateway_session_key = None
    agent._gateway_session_search_profile = None
    result = json.loads(agent._invoke_tool(
        "session_search",
        {"query": "Hermes"},
        "task-id",
    ))

    assert result["success"] is False
    assert "profile boundary is unavailable" in result["error"]
    assert called is False


def test_sequential_tool_dispatch_applies_same_gateway_boundary(
    tmp_path,
    monkeypatch,
):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db", profile_name="default")
    db.create_session(
        "telegram-session",
        source="telegram",
        session_key="agent:coder:telegram:dm:2",
        profile_name="coder",
    )
    captured = {}
    session_search_mod = ModuleType("tools.session_search_tool")

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True})

    session_search_mod.session_search = fake_session_search
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", session_search_mod)

    agent = _make_agent(db, platform="telegram")
    agent._gateway_session_key = "agent:coder:telegram:dm:2"
    tool_call = SimpleNamespace(
        id="search-1",
        function=SimpleNamespace(
            name="session_search",
            arguments=json.dumps({
                "query": "Hermes",
                "profile": "model-controlled",
            }),
        ),
    )
    messages = []
    agent._execute_tool_calls_sequential(
        SimpleNamespace(tool_calls=[tool_call]),
        messages,
        "task-id",
    )

    assert captured["profile"] == "model-controlled"
    assert captured["profile_name"] == "coder"
    assert captured["allow_cross_profile"] is False
    assert json.loads(messages[-1]["content"])["success"] is True
