import base64
import dataclasses
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    _UNATTRIBUTED_SESSION_CONTEXT_METADATA_KEY,
)
from gateway.session import (
    SessionContext,
    SessionSource,
    _hash_chat_id,
    _hash_message_id,
    _hash_sender_id,
)
from gateway.session_context import (
    get_session_env,
    session_redact_pii_enabled,
)


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6L2ioAAAAASUVORK5CYII="
)


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureQueuedNativeImageAgent:
    calls = []
    histories = []
    session_contexts = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        from tools.mcp_tool import _build_session_context_meta

        type(self).calls.append(message)
        type(self).histories.append(conversation_history)
        type(self).session_contexts.append(
            {
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "message_id": get_session_env("HERMES_SESSION_MESSAGE_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
                "redact_pii": session_redact_pii_enabled(),
                "mcp_meta": _build_session_context_meta(),
            }
        )
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


class CaptureUnattributedFollowupAgent:
    calls = []
    session_contexts = []
    followup_key = "interrupt_message"

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        from tools.mcp_tool import _build_session_context_meta

        type(self).calls.append(message)
        type(self).session_contexts.append(
            {
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "message_id": get_session_env("HERMES_SESSION_MESSAGE_ID"),
                "redact_pii": session_redact_pii_enabled(),
                "mcp_meta": _build_session_context_meta(),
            }
        )
        if len(type(self).calls) == 1:
            result = {
                "final_response": "",
                "messages": [],
                "api_calls": 1,
            }
            if type(self).followup_key == "interrupt_message":
                result.update(
                    interrupted=True,
                    interrupt_message="use the fallback instead",
                )
            else:
                result["pending_steer"] = "use the fallback instead"
            return result
        return {
            "final_response": "fallback-done",
            "messages": [],
            "api_calls": 1,
        }


class CapturePendingSteerFollowupAgent(CaptureUnattributedFollowupAgent):
    calls = []
    session_contexts = []
    followup_key = "pending_steer"


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        get_connected_platforms=lambda: [Platform.TELEGRAM],
        get_home_channel=lambda _platform: None,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_kw: "native"
    return runner


@pytest.mark.asyncio
async def test_cross_lane_queued_native_image_is_redispatched_without_recursive_transcript(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.histories = []
    CaptureQueuedNativeImageAgent.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    (tmp_path / "config.yaml").write_text(
        "privacy:\n  redact_pii: true\n",
        encoding="utf-8",
    )

    adapter = CaptureAdapter()
    adapter.handle_message = AsyncMock()
    runner = _make_runner(adapter)
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = True

    image_path = tmp_path / "queued-image.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="first-user",
        message_id="first-message",
    )
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
        user_id="queued-user",
    )

    pending_event = MessageEvent(
        text="describe this",
        message_type=MessageType.PHOTO,
        source=pending_source,
        media_urls=[str(image_path)],
        media_types=["image/png"],
        message_id="queued-1",
    )
    session_key = "agent:main:telegram:group:-1001"
    adapter._pending_messages[session_key] = pending_event

    initial_context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key=session_key,
        session_id="sess-native-image-followup",
    )
    tokens, policy = runner._bind_session_context_for_turn(initial_context)
    assert policy is True
    try:
        result = await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-native-image-followup",
            session_key=session_key,
        )
    finally:
        runner._clear_session_env(tokens)

    assert result["final_response"] == "done-1"
    assert CaptureQueuedNativeImageAgent.calls == ["hello"]
    assert CaptureQueuedNativeImageAgent.histories == [[]]
    assert len(CaptureQueuedNativeImageAgent.session_contexts) == 1
    adapter.handle_message.assert_awaited_once_with(pending_event)
    assert pending_event.source.chat_id == pending_source.chat_id
    assert pending_event.source.thread_id == pending_source.thread_id
    assert pending_event.source.user_id == pending_source.user_id
    assert pending_event.source.message_id == "queued-1"
    assert pending_event.media_urls == [str(image_path)]
    assert pending_event.media_types == ["image/png"]
    first_context = CaptureQueuedNativeImageAgent.session_contexts[0]
    assert first_context["user_id"] == "first-user"
    assert first_context["message_id"] == "first-message"
    assert first_context["redact_pii"] is True
    assert first_context["mcp_meta"]["com.nousresearch.hermes/user_id"] == (
        _hash_sender_id("first-user")
    )
    assert first_context["mcp_meta"]["com.nousresearch.hermes/message_id"] == (
        _hash_message_id("first-message")
    )
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_MESSAGE_ID") == ""
    assert session_redact_pii_enabled() is False


@pytest.mark.asyncio
async def test_cross_lane_queued_voice_is_redispatched_before_transcription_or_echo(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.histories = []
    CaptureQueuedNativeImageAgent.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    adapter = CaptureAdapter()
    adapter.handle_message = AsyncMock()
    runner = _make_runner(adapter)
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = True
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=("private queued transcript", ["private queued transcript"])
    )
    runner._should_echo_stt_transcripts = lambda: True

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="first-user",
        message_id="first-message",
    )
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="private-thread",
        user_id="queued-user",
    )
    pending_event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=pending_source,
        media_urls=[str(tmp_path / "private-voice.ogg")],
        media_types=["audio/ogg"],
        message_id="queued-voice",
    )
    session_key = "agent:main:telegram:group:-1001"
    adapter._pending_messages[session_key] = pending_event

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-cross-lane-voice",
        session_key=session_key,
    )

    assert result["final_response"] == "done-1"
    runner._enrich_message_with_transcription.assert_not_awaited()
    adapter.handle_message.assert_awaited_once_with(pending_event)
    assert pending_event.text == ""
    assert pending_event.media_urls == [str(tmp_path / "private-voice.ogg")]
    assert all(not item["content"].startswith("🎙️") for item in adapter.sent)


@pytest.mark.asyncio
async def test_same_lane_queued_native_image_recurses_with_queued_context(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.histories = []
    CaptureQueuedNativeImageAgent.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    (tmp_path / "config.yaml").write_text(
        "privacy:\n  redact_pii: true\n",
        encoding="utf-8",
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = True

    image_path = tmp_path / "queued-same-lane-image.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="same-user",
        message_id="first-message",
    )
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="same-user",
    )
    session_key = "agent:main:telegram:group:-1001:same-user"
    adapter._pending_messages[session_key] = MessageEvent(
        text="describe this",
        message_type=MessageType.PHOTO,
        source=pending_source,
        media_urls=[str(image_path)],
        media_types=["image/png"],
        message_id="queued-1",
    )

    initial_context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key=session_key,
        session_id="sess-native-image-same-lane",
    )
    tokens, policy = runner._bind_session_context_for_turn(initial_context)
    assert policy is True
    context_prompts = []
    original_run_agent = runner._run_agent

    async def capture_context_prompt(*args, **kwargs):
        context_prompts.append(
            kwargs.get("context_prompt", args[1] if len(args) > 1 else None)
        )
        return await original_run_agent(*args, **kwargs)

    runner._run_agent = capture_context_prompt
    try:
        result = await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-native-image-same-lane",
            session_key=session_key,
        )
    finally:
        runner._clear_session_env(tokens)

    assert result["final_response"] == "done-2"
    assert len(CaptureQueuedNativeImageAgent.calls) == 2
    queued_message = CaptureQueuedNativeImageAgent.calls[1]
    assert isinstance(queued_message, list)
    assert queued_message[0]["type"] == "text"
    assert queued_message[0]["text"].startswith("describe this")
    assert any(part.get("type") == "image_url" for part in queued_message)
    first_context, queued_context = CaptureQueuedNativeImageAgent.session_contexts
    assert first_context["user_id"] == "same-user"
    assert first_context["message_id"] == "first-message"
    assert first_context["redact_pii"] is True
    assert first_context["mcp_meta"]["com.nousresearch.hermes/user_id"] == (
        _hash_sender_id("same-user")
    )
    assert first_context["mcp_meta"]["com.nousresearch.hermes/message_id"] == (
        _hash_message_id("first-message")
    )
    assert queued_context["user_id"] == "same-user"
    assert queued_context["message_id"] == "queued-1"
    assert queued_context["session_key"] == session_key
    assert queued_context["redact_pii"] is True
    assert queued_context["mcp_meta"]["com.nousresearch.hermes/user_id"] == (
        _hash_sender_id("same-user")
    )
    assert queued_context["mcp_meta"]["com.nousresearch.hermes/message_id"] == (
        _hash_message_id("queued-1")
    )
    assert len(context_prompts) == 2
    assert "same-user" not in context_prompts[1]
    assert _hash_sender_id("same-user") in context_prompts[1]
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_MESSAGE_ID") == ""
    assert session_redact_pii_enabled() is False


@pytest.mark.asyncio
async def test_same_lane_queued_turn_refreshes_durable_provenance_and_source_cache(
    monkeypatch, tmp_path
):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    import hermes_state

    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.histories = []
    CaptureQueuedNativeImageAgent.session_contexts = []
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner.adapters[Platform.RELAY] = adapter
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner.session_store = store

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="same-user",
        message_id="first-message",
        transport_profile="primary-owner",
    )
    entry = store.get_or_create_session(source)
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="same-user",
        transport_profile="relay-owner",
        delivered_via_upstream_relay=True,
    )
    pending_event = MessageEvent(
        text="queued through relay",
        source=pending_source,
        message_id="queued-message",
    )
    adapter._pending_messages[entry.session_key] = pending_event

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=entry.session_id,
        session_key=entry.session_key,
    )

    assert result["final_response"] == "done-2"
    durable_source = store.routing_source_snapshot(entry.session_key)
    assert durable_source is not None
    assert durable_source.message_id is None
    assert durable_source.delivered_via_upstream_relay is True
    assert durable_source.transport_profile == "relay-owner"
    cached_source = runner._get_cached_session_source(entry.session_key)
    assert cached_source is not None
    assert cached_source.message_id is None
    assert cached_source.delivered_via_upstream_relay is True
    assert cached_source.transport_profile == "relay-owner"
    store._db.close()


@pytest.mark.asyncio
async def test_same_lane_queued_turn_does_not_overwrite_session_reset(
    monkeypatch, tmp_path
):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    import hermes_state

    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.histories = []
    CaptureQueuedNativeImageAgent.session_contexts = []
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner.session_store = store
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="same-user",
        message_id="first-message",
        transport_profile="primary-owner",
    )
    entry = store.get_or_create_session(source)
    runner._cache_session_source(entry.session_key, source)
    pending_event = MessageEvent(
        text="stale queued turn",
        source=dataclasses.replace(
            source,
            message_id=None,
            transport_profile="stale-owner",
            delivered_via_upstream_relay=True,
        ),
        message_id="stale-queued-message",
    )
    adapter._pending_messages[entry.session_key] = pending_event
    original_refresh = store.refresh_routing_origin_if_current

    def reset_before_refresh(session_key, expected_session_id, queued_source):
        reset = store.reset_session(session_key)
        assert reset is not None
        return original_refresh(session_key, expected_session_id, queued_source)

    monkeypatch.setattr(
        store,
        "refresh_routing_origin_if_current",
        reset_before_refresh,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=entry.session_id,
        session_key=entry.session_key,
    )

    assert result["final_response"] == "done-1"
    assert CaptureQueuedNativeImageAgent.calls == ["hello"]
    assert adapter._pending_messages[entry.session_key] is pending_event
    current = store.routing_source_snapshot(entry.session_key)
    assert current is not None
    assert current.delivered_via_upstream_relay is False
    assert current.transport_profile == "primary-owner"
    cached = runner._get_cached_session_source(entry.session_key)
    assert cached is not None
    assert cached.transport_profile == "primary-owner"
    assert cached.delivered_via_upstream_relay is False
    store._db.close()


@pytest.mark.asyncio
async def test_pinned_queued_event_for_other_session_is_requeued_without_recursing(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.histories = []
    CaptureQueuedNativeImageAgent.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner.config.group_sessions_per_user = True
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="same-user",
        message_id="current-message",
    )
    session_key = "agent:main:telegram:group:-1001:same-user"
    pending_event = MessageEvent(
        text="completion for the old session",
        source=source,
        message_id="completion-message",
        metadata={"gateway_session_id": "sess-old"},
    )
    newer_event = MessageEvent(
        text="newer queued request",
        source=source,
        message_id="newer-message",
    )
    adapter._pending_messages[session_key] = pending_event
    runner._queued_events = {session_key: [newer_event]}
    initial_context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key=session_key,
        session_id="sess-current",
    )
    original_history = [{"role": "user", "content": "prior current-session turn"}]

    tokens = runner._set_session_env(initial_context, redact_pii=True)
    try:
        result = await runner._run_agent(
            message="current request",
            context_prompt="",
            history=original_history,
            source=source,
            session_id="sess-current",
            session_key=session_key,
        )
    finally:
        runner._clear_session_env(tokens)

    assert result["final_response"] == "done-1"
    assert CaptureQueuedNativeImageAgent.calls == ["current request"]
    assert CaptureQueuedNativeImageAgent.histories == [original_history]
    assert len(CaptureQueuedNativeImageAgent.session_contexts) == 1
    assert adapter._pending_messages[session_key] is pending_event
    assert runner._queued_events[session_key] == [newer_event]


@pytest.mark.parametrize(
    "agent_class",
    [CaptureUnattributedFollowupAgent, CapturePendingSteerFollowupAgent],
)
@pytest.mark.asyncio
async def test_string_only_followup_omits_prior_turn_identity(
    monkeypatch, tmp_path, agent_class
):
    agent_class.calls = []
    agent_class.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_class
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    (tmp_path / "config.yaml").write_text(
        "privacy:\n  redact_pii: true\n",
        encoding="utf-8",
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="first-user",
        message_id="first-message",
    )
    initial_context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key="agent:main:telegram:group:-1001",
        session_id="sess-string-followup",
    )
    tokens = runner._set_session_env(initial_context, redact_pii=True)
    context_prompts = []
    original_run_agent = runner._run_agent

    async def capture_context_prompt(*args, **kwargs):
        context_prompts.append(
            kwargs.get("context_prompt", args[1] if len(args) > 1 else None)
        )
        return await original_run_agent(*args, **kwargs)

    runner._run_agent = capture_context_prompt
    try:
        result = await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-string-followup",
            session_key="agent:main:telegram:group:-1001",
        )
    finally:
        runner._clear_session_env(tokens)

    assert result["final_response"] == "fallback-done"
    assert agent_class.calls == [
        "hello",
        "use the fallback instead",
    ]
    first_context, fallback_context = agent_class.session_contexts
    assert first_context["user_id"] == "first-user"
    assert first_context["message_id"] == "first-message"
    assert first_context["mcp_meta"] is not None
    assert fallback_context == {
        "user_id": "",
        "message_id": "",
        "redact_pii": None,
        "mcp_meta": None,
    }
    assert len(context_prompts) == 2
    assert "first-user" not in context_prompts[1]
    assert "-1001" not in context_prompts[1]
    assert _hash_chat_id("-1001") in context_prompts[1]


@pytest.mark.parametrize("has_routing_source", [False, True])
@pytest.mark.asyncio
async def test_unattributed_queued_followup_omits_identity_and_keeps_queue_lane(
    monkeypatch, tmp_path, has_routing_source
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    (tmp_path / "config.yaml").write_text(
        "privacy:\n  redact_pii: true\n",
        encoding="utf-8",
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner.config.group_sessions_per_user = True
    session_key = "agent:main:telegram:group:-1001:first-user"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="first-user",
        message_id="first-message",
    )
    pending_event = MessageEvent(
        text="source-less followup",
        source=source if has_routing_source else None,
        message_id="orphan-message",
        platform_update_id=1234,
        reply_to_message_id="orphan-parent",
        reply_to_text="untrusted reply context",
        reply_to_author_id="orphan-author",
        reply_to_author_name="Orphan Author",
        reply_to_is_own_message=True,
        metadata=(
            {_UNATTRIBUTED_SESSION_CONTEXT_METADATA_KEY: True}
            if has_routing_source
            else {}
        ),
    )
    adapter._pending_messages[session_key] = pending_event
    initial_context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key=session_key,
        session_id="sess-source-less-followup",
    )

    tokens = runner._set_session_env(initial_context, redact_pii=True)
    context_prompts = []
    original_run_agent = runner._run_agent

    async def capture_context_prompt(*args, **kwargs):
        context_prompts.append(
            kwargs.get("context_prompt", args[1] if len(args) > 1 else None)
        )
        return await original_run_agent(*args, **kwargs)

    runner._run_agent = capture_context_prompt
    try:
        result = await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-source-less-followup",
            session_key=session_key,
        )
    finally:
        runner._clear_session_env(tokens)

    assert result["final_response"] == "done-2"
    assert CaptureQueuedNativeImageAgent.calls == ["hello", "source-less followup"]
    first_context, followup_context = CaptureQueuedNativeImageAgent.session_contexts
    assert first_context["user_id"] == "first-user"
    assert first_context["message_id"] == "first-message"
    assert first_context["mcp_meta"] is not None
    assert followup_context == {
        "user_id": "",
        "message_id": "",
        "session_key": session_key,
        "redact_pii": None,
        "mcp_meta": None,
    }
    assert pending_event.message_id is None
    assert pending_event.platform_update_id is None
    assert pending_event.reply_to_message_id is None
    assert pending_event.reply_to_text is None
    assert pending_event.reply_to_author_id is None
    assert pending_event.reply_to_author_name is None
    assert pending_event.reply_to_is_own_message is False
    assert pending_event.metadata[_UNATTRIBUTED_SESSION_CONTEXT_METADATA_KEY] is True
    assert all(delivery["reply_to"] != "orphan-message" for delivery in adapter.sent)
    assert len(context_prompts) == 2
    assert "first-user" not in context_prompts[1]
    assert "-1001" not in context_prompts[1]
    assert _hash_chat_id("-1001") in context_prompts[1]


@pytest.mark.asyncio
async def test_recursion_cap_requeues_media_without_merge_or_reorder(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    CaptureQueuedNativeImageAgent.session_contexts = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner._MAX_INTERRUPT_DEPTH = 0
    runner._queued_events = {}
    session_key = "agent:main:telegram:group:-1001"
    alice_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="alice",
        message_id="alice-message",
    )
    bob_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="bob",
        message_id="bob-message",
    )
    alice = MessageEvent(
        text="alice photo",
        message_type=MessageType.PHOTO,
        source=alice_source,
        media_urls=["/tmp/alice.jpg"],
        media_types=["image/jpeg"],
    )
    bob = MessageEvent(
        text="bob photo",
        message_type=MessageType.PHOTO,
        source=bob_source,
        media_urls=["/tmp/bob.jpg"],
        media_types=["image/jpeg"],
    )
    adapter._pending_messages[session_key] = alice
    runner._queued_events[session_key] = [bob]
    initial_context = SessionContext(
        source=alice_source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key=session_key,
        session_id="sess-recursion-cap",
    )

    tokens = runner._set_session_env(initial_context, redact_pii=True)
    try:
        result = await runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=alice_source,
            session_id="sess-recursion-cap",
            session_key=session_key,
        )
    finally:
        runner._clear_session_env(tokens)

    assert result["final_response"] == "done-1"
    assert CaptureQueuedNativeImageAgent.calls == ["hello"]
    assert adapter._pending_messages[session_key] is alice
    assert alice.text == "alice photo"
    assert alice.media_urls == ["/tmp/alice.jpg"]
    assert runner._queued_events[session_key] == [bob]
