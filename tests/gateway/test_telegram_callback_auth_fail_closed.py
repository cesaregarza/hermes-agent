"""Tests for Telegram adapter fail-closed auth fallback (#24457).

The _is_callback_user_authorized fallback must deny users by default
when TELEGRAM_ALLOWED_USERS is empty, instead of allowing everyone.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform


# -- Fake telegram modules (minimal stubs) --------------------------------

_fake_telegram_error = types.ModuleType("telegram.error")


class _TelegramError(Exception):
    pass


_fake_telegram_error.TelegramError = _TelegramError
_fake_telegram_error.BadRequest = type("BadRequest", (_TelegramError,), {})
_fake_telegram_error.NetworkError = type("NetworkError", (_TelegramError,), {})

_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(HTML="HTML")

_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = type("HTTPXRequest", (), {"__init__": lambda *a, **kw: None})

_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.ApplicationBuilder = type("ApplicationBuilder", (), {
    "token": lambda self, *a: self,
    "build": lambda self: None,
})

_fake_telegram = types.ModuleType("telegram")
_fake_telegram.error = _fake_telegram_error
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram.ext = _fake_telegram_ext
_fake_telegram.request = _fake_telegram_request


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    return adapter


class TestCallbackAuthFailClosed:
    """_is_callback_user_authorized fallback must be fail-closed."""

    def test_no_allowlist_no_allow_all_denies(self, monkeypatch):
        """No TELEGRAM_ALLOWED_USERS and no GATEWAY_ALLOW_ALL_USERS → deny."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()
        # Force the fallback path (no runner auth)
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is False

    def test_no_allowlist_with_global_allow_all_permits(self, monkeypatch):
        """No TELEGRAM_ALLOWED_USERS but GATEWAY_ALLOW_ALL_USERS=true → allow."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is True

    def test_allowlist_with_matching_user_permits(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS contains the user → allow."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345,67890")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is True

    def test_allowlist_without_matching_user_denies(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS does not contain the user → deny."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "67890")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is False

    def test_allowlist_wildcard_permits(self, monkeypatch):
        """TELEGRAM_ALLOWED_USERS=* → allow everyone."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        adapter = _make_adapter()
        adapter._message_handler = None
        assert adapter._is_callback_user_authorized("12345") is True

    def test_secondary_uses_registered_profile_auth_not_primary_env(self, monkeypatch):
        """A closure-bound secondary honors its own allowlist/pairing callback."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "primary-user")
        monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()

        async def secondary_handler(_event):
            return None

        adapter._message_handler = secondary_handler
        adapter._profile_routes_enabled = False
        calls = []

        def profile_auth(user_id, chat_type, chat_id):
            calls.append((user_id, chat_type, chat_id))
            # The profile callback is the same runner path that honors both
            # the secondary allowlist and secondary PairingStore approvals.
            return user_id in {"secondary-user", "paired-user"}

        adapter._authorization_check = profile_auth

        assert adapter._is_callback_user_authorized(
            "secondary-user",
            chat_id="-1001",
            chat_type="supergroup",
            thread_id="42",
        ) is True
        assert adapter._is_callback_user_authorized(
            "paired-user",
            chat_id="-1001",
            chat_type="supergroup",
            thread_id="42",
        ) is True
        assert adapter._is_callback_user_authorized("primary-user") is False
        assert adapter._is_callback_user_authorized("intruder") is False
        assert calls[0] == ("secondary-user", "forum", "-1001")

    def test_secondary_missing_or_failing_profile_auth_denies(self, monkeypatch):
        """Primary allow-all cannot rescue missing/broken secondary auth wiring."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()
        adapter._message_handler = lambda _event: None
        adapter._profile_routes_enabled = False

        adapter._authorization_check = None
        assert adapter._is_callback_user_authorized("intruder") is False

        def broken_auth(*_args):
            raise RuntimeError("secondary auth unavailable")

        adapter._authorization_check = broken_auth
        assert adapter._is_callback_user_authorized("intruder") is False
