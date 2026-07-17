"""Shared profile-evidence classification for persisted session rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional


QUARANTINED_SESSION_PROFILE = "!invalid-profile-evidence"
"""Reserved invalid marker for rows whose profile evidence cannot be trusted."""


@dataclass(frozen=True)
class SessionProfileClassification:
    """Resolved profile boundary, or a fail-closed classification error."""

    profile: Optional[str]
    evidence: tuple[tuple[str, str], ...]
    error: Optional[str] = None

    @property
    def coherent(self) -> bool:
        return self.profile is not None and self.error is None


def _canonical_profile(value: Any) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, validate_profile_name

        profile = normalize_profile_name(value)
        validate_profile_name(profile)
    except (ImportError, TypeError, ValueError):
        return None
    # ``agent:main`` is the default/legacy wire namespace, so a named profile
    # literally called main cannot be represented safely by gateway keys.
    return None if profile == "main" else profile


def _failure(reason: str, evidence: dict[str, str]) -> SessionProfileClassification:
    return SessionProfileClassification(
        profile=None,
        evidence=tuple(evidence.items()),
        error=reason,
    )


def classify_session_profile_evidence(
    row: Mapping[str, Any],
    legacy_owner: Any,
) -> SessionProfileClassification:
    """Resolve one session row's profile without weakening explicit evidence.

    ``profile_name`` and ``origin.profile`` are strong evidence. A named
    ``agent:<profile>:...`` key is also evidence. ``agent:main`` is ambiguous
    for pre-multiplex rows: without stronger evidence it means the persisted DB
    primary, while a strong explicit profile takes precedence. Rows with no
    evidence likewise belong to that primary. Malformed or conflicting fields
    never receive a fallback profile.
    """
    owner = _canonical_profile(legacy_owner)
    evidence: dict[str, str] = {}
    if owner is None:
        return _failure("invalid legacy profile owner", evidence)

    raw_profile = str(row.get("profile_name") or "").strip()
    if raw_profile:
        profile = _canonical_profile(raw_profile)
        if profile is None:
            return _failure("invalid profile_name", evidence)
        evidence["profile_name"] = profile

    persisted_origin = row.get("origin_json")
    if persisted_origin is not None and str(persisted_origin).strip():
        try:
            origin = (
                persisted_origin
                if isinstance(persisted_origin, dict)
                else json.loads(str(persisted_origin))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return _failure("malformed origin_json", evidence)
        if not isinstance(origin, dict):
            return _failure("origin_json is not an object", evidence)
        raw_origin_profile = str(origin.get("profile") or "").strip()
        if raw_origin_profile:
            origin_profile = _canonical_profile(raw_origin_profile)
            if origin_profile is None:
                return _failure("invalid origin.profile", evidence)
            evidence["origin.profile"] = origin_profile

    raw_key = str(row.get("session_key") or "").strip()
    legacy_main_key = False
    if raw_key:
        parts = raw_key.split(":")
        if len(parts) < 2 or parts[0] != "agent" or not parts[1]:
            return _failure("invalid session_key namespace", evidence)
        namespace = parts[1]
        if namespace == "main":
            legacy_main_key = True
        else:
            key_profile = _canonical_profile(namespace)
            # ``default`` is represented only by the historical ``main`` wire
            # literal. Named namespaces are already canonical lowercase IDs.
            # Accepting aliases here would turn malformed persisted routing
            # metadata into trusted profile evidence.
            if (
                key_profile is None
                or key_profile == "default"
                or key_profile != namespace
            ):
                return _failure("invalid session_key profile", evidence)
            evidence["session_key"] = key_profile

    profiles = set(evidence.values())
    if len(profiles) > 1:
        return _failure("conflicting profile evidence", evidence)
    if profiles:
        resolved = next(iter(profiles))
        if legacy_main_key:
            evidence["session_key"] = resolved
        return SessionProfileClassification(resolved, tuple(evidence.items()))

    # A pre-multiplex named-primary row still used agent:main. Treat that key,
    # and a wholly unscoped row, as primary-owned rather than default-owned.
    source = "legacy session_key" if legacy_main_key else "legacy owner"
    evidence[source] = owner
    return SessionProfileClassification(owner, tuple(evidence.items()))
