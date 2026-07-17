"""Regression tests for multiplex profile-aware own-policy authorization."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


@pytest.fixture
def profile_auth_env(tmp_path, monkeypatch):
    """Create real default/coder profile homes without touching user state."""
    home = tmp_path / ".hermes"
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, coder_home


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WECOM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_multiplex_runner(monkeypatch):
    """Runner with default allowlist WeCom and secondary open-policy WeCom."""
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    default_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="allowlist",
        _group_policy="pairing",
    )
    secondary_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="open",
        _group_policy="open",
    )

    runner.adapters = {Platform.WECOM: default_adapter}
    runner._profile_adapters = {
        "coder": {Platform.WECOM: secondary_adapter},
    }
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner, default_adapter, secondary_adapter


def _make_profile_env_runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {}
    return runner


def _telegram_source(profile: str, user_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id=f"dm-{user_id}",
        user_name=user_id,
        chat_type="dm",
        profile=profile,
        transport_profile=profile,
    )


def test_profile_env_allowlists_are_disjoint_and_process_env_is_ignored(
    profile_auth_env,
    monkeypatch,
):
    """Multiplex auth reads only the routed profile's real ``.env`` file."""
    home, coder_home = profile_auth_env
    (home / ".env").write_text(
        "TELEGRAM_ALLOWED_USERS=default-user\n",
        encoding="utf-8",
    )
    (coder_home / ".env").write_text(
        "TELEGRAM_ALLOWED_USERS=coder-user\n",
        encoding="utf-8",
    )
    # Poison the process environment with values that would fail open if the
    # multiplex path inherited os.environ.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "process-user")
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    runner = _make_profile_env_runner()

    assert runner._is_user_authorized(
        _telegram_source("default", "default-user")
    ) is True
    assert runner._is_user_authorized(
        _telegram_source("default", "coder-user")
    ) is False
    assert runner._is_user_authorized(
        _telegram_source("coder", "coder-user")
    ) is True
    assert runner._is_user_authorized(
        _telegram_source("coder", "default-user")
    ) is False
    assert runner._is_user_authorized(
        _telegram_source("coder", "process-user")
    ) is False
    assert runner._is_user_authorized(
        _telegram_source("coder", "stranger")
    ) is False


def test_primary_profile_allow_all_does_not_leak_into_coder(
    profile_auth_env,
):
    """Coder remains allowlisted-only when the primary profile is open."""
    home, coder_home = profile_auth_env
    (home / ".env").write_text(
        "TELEGRAM_ALLOW_ALL_USERS=true\n",
        encoding="utf-8",
    )
    (coder_home / ".env").write_text(
        "TELEGRAM_ALLOWED_USERS=coder-user\n"
        "TELEGRAM_ALLOW_ALL_USERS=false\n",
        encoding="utf-8",
    )
    runner = _make_profile_env_runner()

    assert runner._is_user_authorized(
        _telegram_source("default", "anyone")
    ) is True
    assert runner._is_user_authorized(
        _telegram_source("coder", "coder-user")
    ) is True
    assert runner._is_user_authorized(
        _telegram_source("coder", "anyone")
    ) is False


def test_missing_secondary_pairing_store_denies_instead_of_inheriting_primary(
    profile_auth_env,
    monkeypatch,
):
    """An unavailable coder store cannot reuse a primary CLI approval."""
    home, coder_home = profile_auth_env
    (home / ".env").write_text("", encoding="utf-8")
    (coder_home / ".env").write_text("", encoding="utf-8")
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    runner = _make_profile_env_runner()
    primary_store = MagicMock()
    primary_store.is_approved.return_value = True
    runner.pairing_store = primary_store
    runner.pairing_stores = {"default": primary_store}

    assert runner._is_user_authorized(
        _telegram_source("coder", "primary-approved-user")
    ) is False
    primary_store.is_approved.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_profile", "uses_primary_transport"),
    [("coder", False), ("default", True)],
)
async def test_pairing_issuance_uses_runtime_profile_store_only(
    monkeypatch,
    transport_profile,
    uses_primary_transport,
):
    """Coder pairing stays scoped when delivered directly or by a shared bot."""
    from gateway.run import GatewayRunner

    primary_store = MagicMock()
    coder_store = MagicMock()
    coder_store._is_rate_limited.return_value = False
    coder_store.generate_code.return_value = "CODER123"
    primary_adapter = SimpleNamespace(send=AsyncMock())
    coder_adapter = SimpleNamespace(send=AsyncMock())

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {
        "coder": {Platform.TELEGRAM: coder_adapter},
    }
    runner.pairing_store = primary_store
    runner.pairing_stores = {"coder": coder_store}
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompt_pending = {}
    runner._is_user_authorized = lambda _source: False
    runner._get_unauthorized_dm_behavior = lambda *_args, **_kwargs: "pair"
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [],
    )
    event = MessageEvent(
        text="hello",
        message_id="message-1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="coder-user",
            chat_id="coder-dm",
            user_name="Coder",
            chat_type="dm",
            profile="coder",
            transport_profile=transport_profile,
        ),
    )

    assert await runner._handle_message(event) is None

    coder_store._is_rate_limited.assert_called_once_with(
        "telegram",
        "coder-user",
    )
    coder_store.generate_code.assert_called_once_with(
        "telegram",
        "coder-user",
        "Coder",
    )
    delivery_adapter = primary_adapter if uses_primary_transport else coder_adapter
    other_adapter = coder_adapter if uses_primary_transport else primary_adapter
    delivery_adapter.send.assert_awaited_once()
    pairing_message = delivery_adapter.send.await_args.args[1]
    assert "CODER123" in pairing_message
    assert "`hermes -p coder pairing approve telegram CODER123`" in pairing_message
    primary_store._is_rate_limited.assert_not_called()
    primary_store.generate_code.assert_not_called()
    other_adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_named_primary_default_secondary_pairing_command_is_scoped(monkeypatch):
    """Multiplex default still needs ``-p default`` when coder is primary."""
    from gateway.run import GatewayRunner

    coder_store = MagicMock()
    default_store = MagicMock()
    default_store._is_rate_limited.return_value = False
    default_store.generate_code.return_value = "DEFAULT1"
    coder_adapter = SimpleNamespace(send=AsyncMock())
    default_adapter = SimpleNamespace(send=AsyncMock())

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "coder"
    runner.adapters = {Platform.TELEGRAM: coder_adapter}
    runner._profile_adapters = {
        "default": {Platform.TELEGRAM: default_adapter},
    }
    runner.pairing_store = coder_store
    runner.pairing_stores = {
        "coder": coder_store,
        "default": default_store,
    }
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompt_pending = {}
    runner._is_user_authorized = lambda _source: False
    runner._get_unauthorized_dm_behavior = lambda *_args, **_kwargs: "pair"
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [],
    )
    event = MessageEvent(
        text="hello",
        message_id="message-default",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="default-user",
            chat_id="default-dm",
            user_name="Default User",
            chat_type="dm",
            profile="default",
            transport_profile="default",
        ),
    )

    assert await runner._handle_message(event) is None

    default_store.generate_code.assert_called_once_with(
        "telegram",
        "default-user",
        "Default User",
    )
    default_adapter.send.assert_awaited_once()
    pairing_message = default_adapter.send.await_args.args[1]
    assert "`hermes -p default pairing approve telegram DEFAULT1`" in pairing_message
    coder_store.generate_code.assert_not_called()
    coder_adapter.send.assert_not_awaited()


def test_secondary_open_policy_not_authorized_by_default_allowlist(monkeypatch):
    """Secondary-profile open intake must not inherit default allowlist trust."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="attacker",
        chat_id="dm-chat",
        user_name="attacker",
        chat_type="dm",
        profile="coder",
    )

    assert runner._adapter_dm_policy(Platform.WECOM, profile="coder") == "open"
    assert runner._adapter_dm_policy(Platform.WECOM) == "allowlist"
    assert runner._is_user_authorized(source) is False


def test_default_profile_still_trusts_own_allowlist(monkeypatch):
    """Default-profile allowlist trust is unchanged when profile is unstamped."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="dm-chat",
        user_name="allowed-user",
        chat_type="dm",
        profile=None,
    )

    assert runner._is_user_authorized(source) is True


def test_secondary_allowlist_still_authorized(monkeypatch):
    """Secondary profile with allowlist policy is trusted on its own adapter."""
    runner, _default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    secondary_adapter._dm_policy = "allowlist"

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="dm-chat",
        user_name="allowed-user",
        chat_type="dm",
        profile="coder",
    )

    assert runner._is_user_authorized(source) is True


def test_adapter_for_source_resolves_secondary_profile_adapter(monkeypatch):
    """Ingress adapter lookup must use the stamped profile's adapter map."""
    runner, default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="attacker",
        chat_id="dm-chat",
        user_name="attacker",
        chat_type="dm",
        profile="coder",
    )

    assert runner._adapter_for_source(source) is secondary_adapter
    assert runner._adapter_for_source(
        SessionSource(
            platform=Platform.WECOM,
            user_id="allowed-user",
            chat_id="dm-chat",
            user_name="allowed-user",
            chat_type="dm",
            profile=None,
        )
    ) is default_adapter


def test_secondary_allowlist_dm_behavior_ignores_unauthorized(monkeypatch):
    """Unauthorized-DM behavior must read the secondary adapter's dm_policy."""
    runner, _default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    secondary_adapter._dm_policy = "allowlist"

    assert runner._get_unauthorized_dm_behavior(
        Platform.WECOM,
        profile="coder",
    ) == "ignore"
    assert runner._get_unauthorized_dm_behavior(Platform.WECOM) == "ignore"


def test_adapter_auth_check_stamps_secondary_profile(monkeypatch):
    """The adapter auth-check callback must stamp its own secondary profile.

    Regression for the gap where ``_make_adapter_auth_check`` built a
    profile-less ``SessionSource``, so a secondary adapter's external-context
    authorization (e.g. Slack/Discord thread-reply lookups) silently
    resolved the *active* profile's allowlist scope instead of its own.
    """
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    captured: dict = {}

    def fake_is_user_authorized(source):
        captured["profile"] = source.profile
        return True

    runner._is_user_authorized = fake_is_user_authorized

    check = runner._make_adapter_auth_check(Platform.WECOM, profile_name="coder")
    assert check("some-user", "dm", "dm-chat") is True
    assert captured["profile"] == "coder"


def test_adapter_auth_check_defaults_to_primary_profile(monkeypatch):
    """Primary callbacks pin their runtime when no route overrides it."""
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    captured: dict = {}

    def fake_is_user_authorized(source):
        captured["profile"] = source.profile
        return True

    runner._is_user_authorized = fake_is_user_authorized

    check = runner._make_adapter_auth_check(Platform.WECOM)
    assert check("some-user", "dm", "dm-chat") is True
    assert captured["profile"] == "default"


def test_secondary_open_policy_fails_startup_guard(monkeypatch):
    """Secondary profiles must pass the same open-policy startup guard."""
    from gateway.run import _own_policy_open_startup_violation

    _clear_auth_env(monkeypatch)

    secondary_cfg = GatewayConfig(multiplex_profiles=True)
    secondary_cfg.platforms = {
        Platform.WECOM: PlatformConfig(
            enabled=True,
            extra={"dm_policy": "open"},
        ),
    }

    violation = _own_policy_open_startup_violation(secondary_cfg)
    assert violation is not None
    assert "wecom" in violation
    assert "open policy" in violation
