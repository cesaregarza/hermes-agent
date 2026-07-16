import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, _hash_sender_id


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_prefixes_sender_for_shared_non_thread_group_session():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice] hello"


@pytest.mark.asyncio
async def test_preprocess_keeps_plain_text_for_default_group_sessions():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_text",
    ["privacy:\n  redact_pii: true\n", "privacy: [malformed"],
    ids=["enabled", "policy-unavailable"],
)
async def test_profile_scoped_shared_phone_fallback_never_reaches_agent(
    tmp_path,
    monkeypatch,
    config_text,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(config_text, encoding="utf-8")

    runner = _make_runner(
        GatewayConfig(
            multiplex_profiles=True,
            group_sessions_per_user=False,
        )
    )
    phone = "+15551234567"
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="signal-group-1",
        chat_type="group",
        user_id=phone,
        user_name=phone,
        profile="secondary",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_profile_scoped_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == f"[{_hash_sender_id(phone)}] hello"
    assert result is not None
    assert phone not in result


@pytest.mark.asyncio
async def test_shared_phone_fallback_stays_raw_when_policy_is_explicitly_false():
    runner = _make_runner(
        GatewayConfig(
            group_sessions_per_user=False,
        )
    )
    phone = "+15551234567"
    source = SessionSource(
        platform=Platform.BLUEBUBBLES,
        chat_id="iMessage-group-1",
        chat_type="group",
        user_id=phone,
        user_name=phone,
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        redact_pii_policy=False,
    )

    assert result == f"[{phone}] hello"


@pytest.mark.asyncio
async def test_shared_sender_attribution_is_omitted_if_eligibility_lookup_fails(
    monkeypatch,
):
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="signal-group-1",
        chat_type="group",
        user_id="+15551234567",
        user_name="+15551234567",
    )
    event = MessageEvent(text="hello", source=source)

    def _raise(_platform):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("gateway.run._is_pii_redaction_eligible", _raise)
    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
        redact_pii_policy=True,
    )

    assert result == "hello"
    assert result is not None
    assert source.user_name is not None
    assert source.user_name not in result
