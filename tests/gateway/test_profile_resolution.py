"""Tests for GatewayRunner._resolve_profile_home_for_source — profile resolution logic."""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.run import GatewayRunner
from gateway.profile_routing import ProfileRoute
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent


@pytest.fixture
def mock_runner():
    """Create a minimal mock GatewayRunner with the methods we need."""
    runner = MagicMock(spec=GatewayRunner)
    runner.config = MagicMock(profile_routes=[])
    # Bind the actual methods to the mock
    runner._profile_name_for_source = GatewayRunner._profile_name_for_source.__get__(runner)
    runner._resolve_profile_home_for_source = GatewayRunner._resolve_profile_home_for_source.__get__(runner)
    return runner


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Isolate both profile roots and HERMES_HOME for profile E2E tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def inline_to_thread(monkeypatch):
    """Keep adapter-key tests deterministic without a lingering executor."""
    async def _run_inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _run_inline)


@pytest.fixture
def discord_source():
    """Create a basic Discord SessionSource for testing."""
    return SessionSource(
        platform=MagicMock(value="discord"),
        chat_id="123456",
        guild_id="789",
        thread_id=None,
        parent_chat_id=None,
    )


@pytest.fixture
def telegram_source():
    """Create a basic Telegram SessionSource for testing.

    Telegram (like Slack/Feishu/etc.) has no ``guild_id`` — only ``chat_id``.
    Used to prove profile routing is platform-generic, not Discord-only.
    """
    return SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="-1001234567890",
        guild_id=None,
        thread_id=None,
        parent_chat_id=None,
    )


class TestResolutionOrder:
    """Tests that profile resolution follows the correct priority order."""
    
    def test_source_profile_wins_over_routing(self, mock_runner, discord_source):
        """source.profile should be used even if routing would match."""
        discord_source.profile = "from-source"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    mock_get_dir.return_value = Path("/hermes/profiles/from-source")
                    result = mock_runner._resolve_profile_home_for_source(discord_source)
                    
                    assert result == Path("/hermes/profiles/from-source")
                    mock_get_dir.assert_called_once_with("from-source")
    
    def test_routing_wins_over_active_profile(self, mock_runner, discord_source):
        """When source.profile is empty, routing should win over active profile."""
        discord_source.profile = None
        
        # Mock routing to return a profile
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    mock_get_dir.return_value = Path("/hermes/profiles/routed")
                    
                    # Manually set routing to return a profile
                    mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                    
                    result = mock_runner._resolve_profile_home_for_source(discord_source)
                    
                    assert result == Path("/hermes/profiles/routed")
                    mock_get_dir.assert_called_once_with("routed")
    
    def test_active_profile_fallback(self, mock_runner, discord_source):
        """When source.profile and routing both return None, active profile is used."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/active")
                
                # No routing match
                mock_runner._profile_name_for_source = MagicMock(return_value=None)
                
                result = mock_runner._resolve_profile_home_for_source(discord_source)
                
                assert result == Path("/hermes/profiles/active")
                mock_get_dir.assert_called_once_with("active")
    
    def test_default_fallback_when_no_active(self, mock_runner, discord_source):
        """When even active profile is None, 'default' is used."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value=None):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes")
                
                mock_runner._profile_name_for_source = MagicMock(return_value=None)
                
                result = mock_runner._resolve_profile_home_for_source(discord_source)
                
                assert result == Path("/hermes")
                mock_get_dir.assert_called_once_with("default")


class TestMissingProfileWarning:
    """Tests for warning when a profile doesn't exist on disk."""
    
    def test_nonexistent_profile_warning(self, mock_runner, discord_source, caplog):
        """When source.profile points to a nonexistent profile, log a WARNING."""
        discord_source.profile = "nonexistent"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/nonexistent")
                with patch("hermes_cli.profiles.profile_exists", return_value=False):
                    with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                        with caplog.at_level(logging.WARNING):
                            result = mock_runner._resolve_profile_home_for_source(discord_source)
                            
                            # Should fall back to global HERMES_HOME
                            assert result == Path("/hermes")
                            
                            # Should have logged a warning
                            assert len(caplog.records) == 1
                            assert caplog.records[0].levelname == "WARNING"
                            assert "nonexistent" in caplog.records[0].message
                            assert "does not exist" in caplog.records[0].message
                            assert "discord" in caplog.records[0].message
                            assert "123456" in caplog.records[0].message
    
    def test_nonexistent_routing_profile_warning(self, mock_runner, discord_source, caplog):
        """When routing returns a nonexistent profile, log a WARNING."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/routed")
                with patch("hermes_cli.profiles.profile_exists", return_value=False):
                    with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                        # Routing returns a profile that doesn't exist
                        mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                        
                        with caplog.at_level(logging.WARNING):
                            result = mock_runner._resolve_profile_home_for_source(discord_source)
                            
                            # Should fall back to global HERMES_HOME
                            assert result == Path("/hermes")
                            
                            # Should have logged a warning
                            assert len(caplog.records) == 1
                            assert "routed" in caplog.records[0].message
    
    def test_empty_source_profile_no_warning(self, mock_runner, discord_source, caplog):
        """When source.profile is empty, silent fallback to active profile (no warning)."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/active")
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    with caplog.at_level(logging.WARNING):
                        mock_runner._profile_name_for_source = MagicMock(return_value=None)
                        
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        # Should use active profile
                        assert result == Path("/hermes/profiles/active")
                        
                        # No warnings (active profile exists)
                        assert not any(r.levelname == "WARNING" for r in caplog.records)
    
    def test_existing_profile_no_warning(self, mock_runner, discord_source, caplog):
        """When the profile exists, no warning should be logged."""
        discord_source.profile = "existing"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/existing")
                with patch("hermes_cli.profiles.profile_exists", return_value=True):
                    with caplog.at_level(logging.WARNING):
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        assert result == Path("/hermes/profiles/existing")
                        
                        # No warnings
                        assert not any(r.levelname == "WARNING" for r in caplog.records)


class TestExceptionHandling:
    """Tests for exception handling in profile resolution."""
    
    def test_get_profile_dir_exception_logs_warning(self, mock_runner, discord_source, caplog):
        """When get_profile_dir raises an exception, log a WARNING with context."""
        discord_source.profile = "bad-profile"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir", side_effect=ValueError("Invalid profile name")):
                with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                    with caplog.at_level(logging.WARNING):
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        # Should fall back to global HERMES_HOME
                        assert result == Path("/hermes")
                        
                        # Should have logged a warning with exception info
                        assert len(caplog.records) == 1
                        assert caplog.records[0].levelname == "WARNING"
                        assert "bad-profile" in caplog.records[0].message
                        assert "Failed to resolve profile directory" in caplog.records[0].message
    
    def test_exception_with_no_profile_name(self, mock_runner, discord_source, caplog):
        """Exception when no profile was set should still log a warning."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value=None):
            with patch("hermes_cli.profiles.get_profile_dir", side_effect=RuntimeError("Filesystem error")):
                with patch("hermes_constants.get_hermes_home", return_value=Path("/hermes")):
                    mock_runner._profile_name_for_source = MagicMock(return_value=None)
                    
                    with caplog.at_level(logging.WARNING):
                        result = mock_runner._resolve_profile_home_for_source(discord_source)
                        
                        assert result == Path("/hermes")
                        
                        # Warning should mention "(no profile)"
                        assert "(no profile)" in caplog.records[0].message


class TestRoutingConsultation:
    """Tests that _profile_name_for_source is consulted when source.profile is empty."""
    
    def test_routing_consulted_when_source_profile_empty(self, mock_runner, discord_source):
        """_profile_name_for_source should be called when source.profile is empty."""
        discord_source.profile = None
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/routed")
                
                mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                
                mock_runner._resolve_profile_home_for_source(discord_source)
                
                # Should have called routing
                mock_runner._profile_name_for_source.assert_called_once_with(discord_source)
    
    def test_routing_not_consulted_when_source_profile_set(self, mock_runner, discord_source):
        """_profile_name_for_source should NOT be called when source.profile is set."""
        discord_source.profile = "from-source"
        
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            with patch("hermes_cli.profiles.get_profile_dir") as mock_get_dir:
                mock_get_dir.return_value = Path("/hermes/profiles/from-source")
                
                mock_runner._profile_name_for_source = MagicMock(return_value="routed")
                
                mock_runner._resolve_profile_home_for_source(discord_source)
                
                # Should NOT have called routing
                mock_runner._profile_name_for_source.assert_not_called()


class TestNonDiscordProfileRouting:
    """Profile routing must be platform-generic, not Discord-only.

    Regression coverage for the ``gateway_runner`` injection gap: previously
    only Discord's adapter pre-declared ``gateway_runner``, so only Discord
    ever had ``build_source`` call ``_profile_name_for_source``. Telegram /
    Feishu / Slack / etc. silently fell through to the default profile. These
    tests pin the resolution half for a non-Discord platform (Telegram).
    """

    def test_telegram_route_resolves(self, mock_runner, telegram_source):
        """A configured Telegram route resolves to its profile via the real
        ``_profile_name_for_source`` (bound onto the mock runner)."""
        mock_runner.config.profile_routes = [
            ProfileRoute(name="tg", platform="telegram", profile="tg-profile",
                         chat_id="-1001234567890"),
        ]
        telegram_source.profile = None

        assert mock_runner._profile_name_for_source(telegram_source) == "tg-profile"

    def test_telegram_no_route_returns_none(self, mock_runner, telegram_source):
        """With no matching Telegram route, resolution returns None (caller
        falls back to the default/active profile)."""
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="dc-profile",
                         chat_id="123456"),
        ]
        telegram_source.profile = None

        assert mock_runner._profile_name_for_source(telegram_source) is None


class TestGatewayRunnerInjection:
    """``BasePlatformAdapter`` declares ``gateway_runner`` so the gateway's
    unconditional injection reaches every platform adapter — the foundation
    that makes the routing in TestNonDiscordProfileRouting reachable at runtime.
    """

    def test_base_adapter_declares_gateway_runner(self):
        from gateway.platforms.base import BasePlatformAdapter

        # Class-level attribute exists and defaults to None.
        assert hasattr(BasePlatformAdapter, "gateway_runner")
        assert BasePlatformAdapter.gateway_runner is None

    def test_subclass_inherits_gateway_runner(self):
        from gateway.platforms.base import BasePlatformAdapter

        class _ToyAdapter(BasePlatformAdapter):
            pass

        # No manual declaration — yet the attribute is inherited from the base,
        # so the gateway's ``adapter.gateway_runner = self`` injection reaches
        # every adapter, not just the ones that pre-declared it (Discord).
        assert hasattr(_ToyAdapter, "gateway_runner")
        assert _ToyAdapter.gateway_runner is None

    def test_real_signal_primary_receives_runner_and_applies_profile_route(self):
        """Built-in adapters bypass the plugin injection branch but still route."""
        from gateway.config import GatewayConfig
        from gateway.platforms.signal import SignalAdapter

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.config.profile_routes = [
            ProfileRoute(
                name="signal-ops",
                platform="signal",
                profile="ops",
                chat_id="signal-chat",
            )
        ]
        runner._primary_profile_name = "default"
        config = PlatformConfig(
            enabled=True,
            extra={
                "http_url": "http://127.0.0.1:8080",
                "account": "+15551234567",
            },
        )

        adapter = runner._create_primary_adapter(Platform.SIGNAL, config)
        source = adapter.build_source(
            chat_id="signal-chat",
            chat_type="dm",
            user_id="signal-user",
        )

        assert isinstance(adapter, SignalAdapter)
        assert adapter.gateway_runner is runner
        assert source.profile == "ops"
        assert source.transport_profile == "default"


# A concrete adapter we can instantiate without the full platform stack.
# ``build_source`` only reads ``self.platform`` and ``self.gateway_runner``, so a
# bare instance with those two attrs exercises the real BasePlatformAdapter
# method end-to-end. Clearing ``__abstractmethods__`` lets ``__new__`` bypass
# the ABC instantiation guard without stubbing connect/send/get_chat_info/…
class _StubAdapter(BasePlatformAdapter):
    pass


_StubAdapter.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]


def _stub_adapter(platform: Platform, runner) -> "_StubAdapter":
    a = _StubAdapter.__new__(_StubAdapter)
    a.platform = platform
    a.gateway_runner = runner
    return a


class TestAdapterToSessionKeyIntegration:
    """Adapter -> ``source.profile`` -> session-key integration coverage.

    The review asked for integration coverage for Discord AND a non-Discord
    platform. These drive a concrete adapter's real ``build_source``
    (BasePlatformAdapter) with an injected ``gateway_runner``, assert the
    matched route's profile is stamped on the source, and that the resulting
    session key is profile-scoped (``agent:<profile>:...`` rather than the
    shared ``agent:main:...``). The Telegram case is the bug-#2 regression:
    pre-fix it never received ``gateway_runner`` and fell through to default.
    """

    @staticmethod
    def _routes():
        return [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="111", chat_id="222"),
            ProfileRoute(name="tg", platform="telegram", profile="ops",
                         chat_id="-1001234567890"),
        ]

    def test_discord_adapter_stamps_profile_and_scopes_key(self, mock_runner):
        mock_runner.config.profile_routes = self._routes()
        adapter = _stub_adapter(Platform.DISCORD, mock_runner)

        source = adapter.build_source(
            chat_id="222", chat_type="group", guild_id="111", user_id="u1",
        )
        assert source.profile == "coder"

        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:coder:"), key
        # A default-profile key would land in agent:main — must differ.
        assert key != build_session_key(source, profile=None)

    def test_telegram_adapter_stamps_profile_and_scopes_key(self, mock_runner):
        """Non-Discord platform (bug #2). The adapter now receives
        ``gateway_runner``, so ``build_source`` stamps the profile and the
        session key is isolated under ``agent:ops:`` instead of ``agent:main:``."""
        mock_runner.config.profile_routes = self._routes()
        adapter = _stub_adapter(Platform.TELEGRAM, mock_runner)

        source = adapter.build_source(
            chat_id="-1001234567890", chat_type="group", user_id="u1",
        )
        assert source.profile == "ops"

        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:ops:"), key
        assert key != build_session_key(source, profile=None)

    def test_adapter_without_runner_falls_back_to_default_namespace(self, mock_runner):
        """Regression anchor: with no ``gateway_runner`` injected (the pre-fix
        state for non-Discord adapters), ``build_source`` leaves ``profile=None``
        and the session key is the shared ``agent:main:`` namespace — no
        per-profile isolation. This is the silent fallback the fix removes for
        non-Discord platforms."""
        adapter = _stub_adapter(Platform.TELEGRAM, runner=None)

        source = adapter.build_source(
            chat_id="-1001234567890", chat_type="group", user_id="u1",
        )
        assert source.profile is None
        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:main:"), key

    def test_primary_route_separates_runtime_from_transport_owner(self):
        """A shared primary bot may route runtime without changing egress bot."""
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.config.profile_routes = [
            ProfileRoute(
                name="ops-chat",
                platform="telegram",
                profile="ops",
                chat_id="shared-chat",
            )
        ]
        runner._primary_profile_name = "default"
        runner._profile_adapters = {}

        primary = _stub_adapter(Platform.TELEGRAM, runner)
        primary._profile_name = "default"
        primary._profile_routes_enabled = True
        runner.adapters = {Platform.TELEGRAM: primary}

        source = primary.build_source(
            chat_id="shared-chat",
            chat_type="group",
            user_id="operator",
        )

        assert source.profile == "ops"
        assert source.transport_profile == "default"
        assert runner._adapter_for_source(source) is primary
        assert build_session_key(source, profile=source.profile).startswith(
            "agent:ops:"
        )

    def test_secondary_owner_disables_shared_bot_routes(self):
        """A credential-specific adapter pins both runtime and transport owner."""
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.config.profile_routes = [
            ProfileRoute(
                name="ops-chat",
                platform="telegram",
                profile="ops",
                chat_id="shared-chat",
            )
        ]
        runner._primary_profile_name = "default"
        runner.adapters = {Platform.TELEGRAM: object()}

        secondary = _stub_adapter(Platform.TELEGRAM, runner)
        secondary._profile_name = "coder"
        secondary._profile_routes_enabled = False
        runner._profile_adapters = {
            "coder": {Platform.TELEGRAM: secondary},
        }

        source = secondary.build_source(
            chat_id="shared-chat",
            chat_type="group",
            user_id="developer",
        )

        assert source.profile == "coder"
        assert source.transport_profile == "coder"
        assert runner._adapter_for_source(source) is secondary
        assert build_session_key(source, profile=source.profile).startswith(
            "agent:coder:"
        )

    @staticmethod
    def _active_guard_adapter(*, owning_profile: str | None = None):
        adapter = _stub_adapter(Platform.TELEGRAM, runner=None)
        adapter.config = PlatformConfig(enabled=True, token="test")
        adapter._profile_name = owning_profile
        adapter._message_handler = AsyncMock(return_value=None)
        adapter._topic_recovery_fn = None
        adapter._active_sessions = {}
        adapter._session_tasks = {}
        adapter._pending_messages = {}
        adapter._text_debounce = {}
        adapter._text_debounce_overflow = {}
        started: list[str] = []
        busy: list[tuple[MessageEvent, str]] = []

        def _start(event, session_key, *, interrupt_event=None):
            started.append(session_key)
            adapter._active_sessions[session_key] = interrupt_event or asyncio.Event()
            return True

        async def _busy(event, session_key):
            busy.append((event, session_key))
            return True

        adapter._start_session_processing = _start
        adapter._busy_session_handler = _busy
        return adapter, started, busy

    @pytest.mark.asyncio
    async def test_active_guard_isolated_by_stamped_profile(self, inline_to_thread):
        adapter, started, busy = self._active_guard_adapter()
        alpha = MessageEvent(
            text="alpha",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="shared-chat",
                user_id="same-user",
                profile="alpha",
            ),
        )
        beta = MessageEvent(
            text="beta",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="shared-chat",
                user_id="same-user",
                profile="beta",
            ),
        )

        await adapter.handle_message(alpha)
        await adapter.handle_message(beta)

        assert started == [
            "agent:alpha:telegram:dm:shared-chat",
            "agent:beta:telegram:dm:shared-chat",
        ]
        assert busy == []

    @pytest.mark.asyncio
    async def test_secondary_owner_profile_stamped_before_busy_guard(
        self,
        inline_to_thread,
    ):
        adapter, started, busy = self._active_guard_adapter(owning_profile="coder")

        def _event(text: str) -> MessageEvent:
            return MessageEvent(
                text=text,
                source=SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id="secondary-chat",
                    user_id="same-user",
                ),
            )

        first = _event("first")
        followup = _event("followup")
        await adapter.handle_message(first)
        await adapter.handle_message(followup)

        key = "agent:coder:telegram:dm:secondary-chat"
        assert first.source.profile == "coder"
        assert followup.source.profile == "coder"
        assert first.source.transport_profile == "coder"
        assert followup.source.transport_profile == "coder"
        assert started == [key]
        assert busy == [(followup, key)]

    def test_secondary_owner_profile_stamped_by_build_source(self):
        adapter = _stub_adapter(Platform.TELEGRAM, runner=None)
        adapter._profile_name = "coder"

        source = adapter.build_source(
            chat_id="secondary-chat", chat_type="dm", user_id="same-user",
        )

        assert source.profile == "coder"
        assert source.transport_profile == "coder"
        assert build_session_key(source, profile=source.profile).startswith(
            "agent:coder:"
        )

    @pytest.mark.asyncio
    async def test_named_active_primary_guard_matches_session_store_key(
        self,
        inline_to_thread,
    ):
        """No-route traffic uses the named active profile end to end."""
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_name_for_source = (
            GatewayRunner._profile_name_for_source.__get__(runner)
        )
        store = SessionStore.__new__(SessionStore)
        store.config = runner.config
        runner.session_store = store
        adapter, started, busy = self._active_guard_adapter()
        adapter.gateway_runner = runner
        runner._create_adapter = lambda _platform, _config: adapter

        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="coder",
        ):
            primary = runner._create_primary_adapter(
                Platform.TELEGRAM,
                adapter.config,
            )
            source = primary.build_source(
                chat_id="active-chat",
                chat_type="dm",
                user_id="same-user",
            )
            expected_key = store._generate_session_key(source)
            await primary.handle_message(
                MessageEvent(text="hello", source=source)
            )

        assert source.profile == "coder"
        assert source.transport_profile == "coder"
        assert expected_key == "agent:coder:telegram:dm:active-chat"
        assert started == [expected_key]
        assert busy == []


class TestStrictMultiplexProfileResolution:
    @pytest.mark.asyncio
    async def test_unknown_explicit_profile_never_enters_agent(
        self,
        profile_env,
    ):
        """An unknown routed namespace cannot inherit primary credentials."""
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._run_agent_inner = AsyncMock(return_value={"final_response": "unsafe"})
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="unknown-profile-chat",
            user_id="attacker",
            profile="does-not-exist",
            transport_profile="default",
        )

        with pytest.raises(LookupError, match="does-not-exist"):
            await runner._run_agent(
                "hello",
                "context",
                [],
                source,
                "session-unknown",
                session_key="agent:does-not-exist:telegram:dm:unknown-profile-chat",
            )

        runner._run_agent_inner.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_background_profile_never_enters_agent(
        self,
        profile_env,
    ):
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._run_background_task_inner = AsyncMock()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="unknown-background-profile",
            user_id="attacker",
            profile="does-not-exist",
            transport_profile="default",
        )

        with pytest.raises(LookupError, match="does-not-exist"):
            await runner._run_background_task(
                "background prompt",
                source,
                "task-unknown",
            )

        runner._run_background_task_inner.assert_not_awaited()


class TestProfileIdentityValidationBoundary:
    def test_wire_source_profile_is_canonicalized(self):
        source = SessionSource.from_dict(
            {
                "platform": "telegram",
                "chat_id": "wire-profile",
                "profile": "Coder",
            }
        )

        assert source.profile == "coder"

    @pytest.mark.parametrize("profile", ["main", "../escape", "root"])
    def test_wire_source_rejects_reserved_or_unsafe_profile(self, profile):
        with pytest.raises(ValueError):
            SessionSource.from_dict(
                {
                    "platform": "telegram",
                    "chat_id": "wire-profile",
                    "profile": profile,
                }
            )

    @pytest.mark.asyncio
    async def test_base_adapter_rejects_invalid_profile_before_session_guard(
        self,
        inline_to_thread,
    ):
        adapter, started, busy = (
            TestAdapterToSessionKeyIntegration._active_guard_adapter()
        )
        event = MessageEvent(
            text="unsafe",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="profile-boundary",
                user_id="user-1",
                profile="../../escape",
            ),
        )

        await adapter.handle_message(event)

        assert started == []
        assert busy == []
        adapter._message_handler.assert_not_awaited()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("profile", "../escape"),
            ("profile", "main"),
            ("transport_profile", "root"),
            ("transport_profile", "../../primary"),
        ],
    )
    @pytest.mark.asyncio
    async def test_runner_rejects_invalid_profile_before_authorization(
        self,
        field,
        value,
    ):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._is_user_authorized = MagicMock(return_value=True)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="profile-boundary",
            user_id="user-1",
        )
        setattr(source, field, value)

        result = await runner._handle_message(
            MessageEvent(text="hello", source=source)
        )

        assert result is None
        runner._is_user_authorized.assert_not_called()

    def test_session_key_boundary_canonicalizes_both_profile_identities(self):
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="canonical-profile",
            user_id="user-1",
            profile="Coder",
            transport_profile="Default",
        )

        key = runner._session_key_for_source(source)

        assert key == "agent:coder:telegram:dm:canonical-profile"
        assert source.profile == "coder"
        assert source.transport_profile == "default"

    @pytest.mark.parametrize("profile", ["main", "../coder", "root"])
    def test_session_key_boundary_rejects_reserved_or_unsafe_profile(self, profile):
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="invalid-profile",
            user_id="user-1",
            profile=profile,
        )

        with pytest.raises(ValueError):
            runner._session_key_for_source(source)


class TestMultiplexGate:
    """``profile_routes`` only activates under ``gateway.multiplex_profiles``.

    Routing stamps ``source.profile``, which namespaces session/batch keys —
    but the profile-scoped agent run (``_profile_runtime_scope``) only engages
    when multiplexing is on. Without the gate, a configured route with
    multiplexing off would split batch/session keys into ``agent:<profile>``
    while the agent still served the turn from ``agent:main``'s home.
    """

    def test_routes_ignored_when_multiplex_off(self, mock_runner, discord_source):
        mock_runner.config.multiplex_profiles = False
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="789", chat_id="123456"),
        ]
        discord_source.profile = None

        assert mock_runner._profile_name_for_source(discord_source) is None

    def test_routes_active_when_multiplex_on(self, mock_runner, discord_source):
        mock_runner.config.multiplex_profiles = True
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="789", chat_id="123456"),
        ]
        discord_source.profile = None

        assert mock_runner._profile_name_for_source(discord_source) == "coder"

    def test_build_source_leaves_profile_none_when_multiplex_off(self, mock_runner):
        """End-to-end through the real adapter ``build_source``: with routes
        configured but multiplexing off, no profile is stamped and the session
        key stays in the legacy ``agent:main`` namespace — byte-identical to a
        gateway with no routes at all."""
        mock_runner.config.multiplex_profiles = False
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="111", chat_id="222"),
        ]
        adapter = _stub_adapter(Platform.DISCORD, mock_runner)

        source = adapter.build_source(
            chat_id="222", chat_type="group", guild_id="111", user_id="u1",
        )
        assert source.profile is None
        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:main:"), key
