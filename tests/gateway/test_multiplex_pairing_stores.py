"""Regression: per-profile PairingStore creation in _start_secondary_profile_adapters.

``gateway/run.py`` referenced ``PairingStore`` at method scope in
``_start_secondary_profile_adapters`` while the class's only import was
method-local inside ``__init__`` — a ``NameError`` at runtime, silently
swallowed by the enclosing ``try/except``, so multiplexing gateways never
created per-profile pairing stores and authz pairing checks for secondary
profiles fell through to the global whitelist.

These tests drive the REAL method (bound onto a bare runner) with the
profile-enumeration and adapter-startup collaborators stubbed, and assert
the per-profile stores actually materialize.
"""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _bare_runner(
    multiplex: bool = True,
    *,
    primary_profile: str = "default",
    primary_store=None,
):
    runner = object.__new__(GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=multiplex)
    runner._primary_profile_name = primary_profile
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_store = primary_store if primary_store is not None else MagicMock()
    runner.pairing_stores = {}
    return runner


def test_secondary_profile_pairing_stores_created(tmp_path, monkeypatch):
    """The served-profiles loop must create a PairingStore per profile.

    Pre-fix this silently did nothing: the ``PairingStore(profile=name)``
    reference raised NameError inside the swallowed try/except.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
        ("coder", tmp_path / ".hermes" / "profiles" / "coder"),
    ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner._profile_adapters["coder"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    # Both the active profile and the served secondary get a store.
    assert "default" in runner.pairing_stores, (
        "active profile PairingStore missing — the NameError swallow is back"
    )
    assert "coder" in runner.pairing_stores, (
        "secondary profile PairingStore missing — the NameError swallow is back"
    )


def test_pairing_store_scoped_to_profile_dir(tmp_path, monkeypatch):
    """The created store must live under the profile's pairing directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
        ("ops", tmp_path / ".hermes" / "profiles" / "ops"),
    ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner._profile_adapters["ops"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    store = runner.pairing_stores["ops"]
    assert store.profile == "ops"
    assert "profiles/ops/platforms/pairing" in str(store._dir).replace("\\", "/"), (
        f"store not profile-scoped: {store._dir}"
    )


def test_default_primary_reuses_cli_pairing_store(tmp_path, monkeypatch):
    """The default primary map entry is the exact CLI-visible global store."""
    from gateway.pairing import PairingStore

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch("gateway.pairing.PAIRING_DIR", home / "platforms" / "pairing"):
        cli_store = PairingStore()
    cli_store._approve_user("telegram", "cli-approved", "Owner")

    runner = _bare_runner(primary_store=cli_store)

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None
    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("default", home)],
    ):
        asyncio.run(runner._start_secondary_profile_adapters())

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="dm-cli-approved",
        chat_type="dm",
        user_id="cli-approved",
        profile="default",
    )
    assert runner.pairing_stores["default"] is cli_store
    assert runner._pairing_store_for(source).is_approved("telegram", "cli-approved")


def test_named_primary_reuses_cli_store_without_double_nesting(tmp_path, monkeypatch):
    """A named primary aliases its global store while default resolves at root."""
    from gateway.pairing import PairingStore

    home = tmp_path / ".hermes"
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(coder_home))
    with patch(
        "gateway.pairing.PAIRING_DIR",
        coder_home / "platforms" / "pairing",
    ):
        coder_cli_store = PairingStore()

    runner = _bare_runner(
        primary_profile="coder",
        primary_store=coder_cli_store,
    )

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None
    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("default", home), ("coder", coder_home)],
    ):
        asyncio.run(runner._start_secondary_profile_adapters())

    assert runner.pairing_stores["coder"] is coder_cli_store
    assert runner.pairing_stores["default"]._dir == home / "platforms" / "pairing"
    assert "profiles/coder/profiles" not in str(coder_cli_store._dir).replace("\\", "/")


def test_failed_secondary_store_is_missing_and_never_inherits_primary(
    tmp_path,
    monkeypatch,
):
    """A store construction failure leaves coder pairing authorization denied."""
    home = tmp_path / ".hermes"
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    primary_store = MagicMock()
    primary_store.is_approved.return_value = True
    runner = _bare_runner(primary_store=primary_store)

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None
    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("default", home), ("coder", coder_home)],
    ), patch("gateway.pairing.PairingStore", side_effect=OSError("read-only")):
        asyncio.run(runner._start_secondary_profile_adapters())

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="coder-dm",
        chat_type="dm",
        user_id="intruder",
        profile="coder",
    )
    assert "coder" not in runner.pairing_stores
    assert runner._pairing_store_for(source) is None
    primary_store.is_approved.assert_not_called()


def test_missing_named_primary_store_fails_closed(tmp_path, monkeypatch):
    """A named primary with no CLI store does not inherit a sibling store."""
    home = tmp_path / ".hermes"
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(coder_home))
    runner = _bare_runner(primary_profile="coder")
    runner.pairing_store = None
    runner.pairing_stores = {"default": MagicMock()}

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None
    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("coder", coder_home)],
    ):
        asyncio.run(runner._start_secondary_profile_adapters())

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="coder-dm",
        chat_type="dm",
        user_id="coder-user",
        profile="coder",
    )
    assert "coder" not in runner.pairing_stores
    assert runner._pairing_store_for(source) is None
