"""Multiplex session ownership boundaries for the dashboard APIs."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException


@pytest.fixture
def multiplex_dashboard(_isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_constants import get_hermes_home

    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    coder_home = profiles_mod.get_profile_dir("coder")
    coder_home.mkdir(parents=True, exist_ok=True)

    db = hermes_state.SessionDB(profile_name="default")
    try:
        db.create_session(
            session_id="default-owned",
            source="cli",
            session_key="agent:main:cli:default",
            profile_name="default",
        )
        db.append_message(
            "default-owned",
            role="user",
            content="default-only transcript",
        )
        db.create_session(
            session_id="coder-owned",
            source="telegram",
            session_key="agent:coder:telegram:chat",
            profile_name="coder",
        )
        db.append_message(
            "coder-owned",
            role="user",
            content="codersupersecret transcript",
        )

        # Strong fields disagree: this row must be invisible to every profile.
        db.create_session(
            session_id="conflicting-evidence",
            source="telegram",
            session_key="agent:coder:telegram:conflict",
            profile_name="default",
        )
        db._conn.execute(
            "UPDATE sessions SET origin_json = ? WHERE id = ?",
            (json.dumps({"profile": "default"}), "conflicting-evidence"),
        )

        # Migration quarantine markers must never be treated as a real owner.
        db.create_session(
            session_id="quarantined-evidence",
            source="telegram",
            profile_name="default",
        )
        db._conn.execute(
            "UPDATE sessions SET profile_name = ? WHERE id = ?",
            ("!invalid-profile-evidence", "quarantined-evidence"),
        )
    finally:
        db.close()

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    yield client


def test_profile_lists_search_stats_and_aggregate_are_owner_scoped(
    multiplex_dashboard,
):
    client = multiplex_dashboard

    default_rows = client.get(
        "/api/sessions",
        params={"profile": "default", "limit": 20, "min_messages": 0},
    )
    assert default_rows.status_code == 200
    assert {row["id"] for row in default_rows.json()["sessions"]} == {
        "default-owned"
    }

    coder_rows = client.get(
        "/api/sessions",
        params={"profile": "coder", "limit": 20, "min_messages": 0},
    )
    assert coder_rows.status_code == 200
    assert {row["id"] for row in coder_rows.json()["sessions"]} == {
        "coder-owned"
    }
    assert coder_rows.json()["sessions"][0]["profile"] == "coder"

    default_search = client.get(
        "/api/sessions/search",
        params={"profile": "default", "q": "codersupersecret"},
    )
    assert default_search.status_code == 200
    assert default_search.json()["results"] == []

    coder_search = client.get(
        "/api/sessions/search",
        params={"profile": "coder", "q": "codersupersecret"},
    )
    assert coder_search.status_code == 200
    assert [row["session_id"] for row in coder_search.json()["results"]] == [
        "coder-owned"
    ]

    default_stats = client.get(
        "/api/sessions/stats",
        params={"profile": "default"},
    ).json()
    coder_stats = client.get(
        "/api/sessions/stats",
        params={"profile": "coder"},
    ).json()
    assert (default_stats["total"], default_stats["messages"]) == (1, 1)
    assert (coder_stats["total"], coder_stats["messages"]) == (1, 1)

    aggregate = client.get(
        "/api/profiles/sessions",
        params={"profile": "all", "limit": 20, "min_messages": 0},
    )
    assert aggregate.status_code == 200
    rows = aggregate.json()["sessions"]
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {"default-owned", "coder-owned"}
    assert by_id["default-owned"]["profile"] == "default"
    assert by_id["coder-owned"]["profile"] == "coder"
    assert aggregate.json()["profile_totals"] == {"default": 1, "coder": 1}


def test_default_profile_cannot_read_or_mutate_coder_session(
    multiplex_dashboard,
):
    client = multiplex_dashboard

    assert client.get(
        "/api/sessions/coder-owned",
        params={"profile": "default"},
    ).status_code == 404
    assert client.get(
        "/api/sessions/coder-owned/messages",
        params={"profile": "default"},
    ).status_code == 404
    assert client.get(
        "/api/sessions/coder-owned/export",
        params={"profile": "default"},
    ).status_code == 404
    assert client.get(
        "/api/sessions/coder-owned/latest-descendant",
        params={"profile": "default"},
    ).status_code == 404
    assert client.patch(
        "/api/sessions/coder-owned",
        json={"profile": "default", "title": "stolen"},
    ).status_code == 404

    deleted = client.delete(
        "/api/sessions/coder-owned",
        params={"profile": "default"},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True, "already_absent": True}

    coder_detail = client.get(
        "/api/sessions/coder-owned",
        params={"profile": "coder"},
    )
    assert coder_detail.status_code == 200
    assert coder_detail.json()["profile"] == "coder"
    exported = client.get(
        "/api/sessions/coder-owned/export",
        params={"profile": "coder"},
    )
    assert exported.status_code == 200
    assert exported.json()["messages"][0]["content"] == "codersupersecret transcript"

    renamed = client.patch(
        "/api/sessions/coder-owned",
        json={"profile": "coder", "title": "Coder chat"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Coder chat"

    from hermes_state import SessionDB

    db = SessionDB(profile_name="default")
    try:
        db.create_session(
            session_id="default-compression-parent",
            source="cli",
            profile_name="default",
        )
        db.end_session("default-compression-parent", "compression")
        db.create_session(
            session_id="coder-compression-child",
            source="cli",
            profile_name="coder",
            session_key="agent:coder:cli:compression",
            parent_session_id="default-compression-parent",
        )
        db.create_session(
            session_id="default-delete-parent",
            source="cli",
            profile_name="default",
        )
        db.create_session(
            session_id="coder-delete-child",
            source="cli",
            profile_name="coder",
            session_key="agent:coder:cli:delete",
            parent_session_id="default-delete-parent",
        )
    finally:
        db.close()

    archived = client.patch(
        "/api/sessions/default-compression-parent",
        json={"profile": "default", "archived": True},
    )
    assert archived.status_code == 200
    db = SessionDB(profile_name="default")
    try:
        assert db.get_session("default-compression-parent")["archived"] == 1
        assert db.get_session("coder-compression-child")["archived"] == 0
    finally:
        db.close()

    rejected_delete = client.delete(
        "/api/sessions/default-delete-parent",
        params={"profile": "default"},
    )
    assert rejected_delete.status_code == 409
    db = SessionDB(profile_name="default")
    try:
        assert db.get_session("default-delete-parent") is not None
        child = db.get_session("coder-delete-child")
        assert child is not None
        assert child["parent_session_id"] == "default-delete-parent"
    finally:
        db.close()


def test_multiplex_analytics_and_cron_runs_use_durable_owner(
    _isolate_hermes_home,
):
    import hermes_state
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_server import (
        _get_models_analytics,
        _get_usage_analytics,
        _list_cron_job_runs_sync,
    )
    from hermes_constants import get_hermes_home

    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    profiles_mod.get_profile_dir("coder").mkdir(parents=True, exist_ok=True)

    db = hermes_state.SessionDB(profile_name="default")
    try:
        db.create_session(
            session_id="default-analytics",
            source="cli",
            model="default/model",
            profile_name="default",
        )
        db.update_token_counts(
            "default-analytics",
            input_tokens=11,
            output_tokens=7,
        )
        db.create_session(
            session_id="coder-analytics",
            source="cli",
            model="coder/model",
            session_key="agent:coder:cli:analytics",
            profile_name="coder",
        )
        db.update_token_counts(
            "coder-analytics",
            input_tokens=101,
            output_tokens=13,
        )
        db.create_session(
            session_id="cron_shared-job_default",
            source="cron",
            model="default/model",
            profile_name="default",
        )
        db.create_session(
            session_id="cron_shared-job_coder",
            source="cron",
            model="coder/model",
            session_key="agent:coder:cron:shared-job",
            profile_name="coder",
        )
    finally:
        db.close()

    default_usage = _get_usage_analytics(days=7, profile="default")
    coder_usage = _get_usage_analytics(days=7, profile="coder")
    assert default_usage["totals"]["total_input"] == 11
    assert coder_usage["totals"]["total_input"] == 101
    assert {row["model"] for row in default_usage["by_model"]} == {
        "default/model"
    }
    assert {row["model"] for row in coder_usage["by_model"]} == {
        "coder/model"
    }

    default_models = _get_models_analytics(days=7, profile="default")
    coder_models = _get_models_analytics(days=7, profile="coder")
    assert {row["model"] for row in default_models["models"]} == {
        "default/model"
    }
    assert {row["model"] for row in coder_models["models"]} == {
        "coder/model"
    }

    default_runs = _list_cron_job_runs_sync(
        "shared-job",
        profile="default",
    )
    coder_runs = _list_cron_job_runs_sync(
        "shared-job",
        profile="coder",
    )
    assert [row["id"] for row in default_runs["runs"]] == [
        "cron_shared-job_default"
    ]
    assert [row["id"] for row in coder_runs["runs"]] == [
        "cron_shared-job_coder"
    ]


def test_duplicate_rename_leaves_authoritative_row_unchanged_on_physical_failure(
    monkeypatch,
):
    from hermes_cli import web_server

    calls: list[str] = []

    class _DB:
        def __init__(self, name: str, *, reject: bool = False):
            self.name = name
            self.reject = reject
            self.title = "Original"

        def set_session_title(self, session_id, title, *, profile_name):
            calls.append(self.name)
            assert session_id == "duplicate"
            assert profile_name == "coder"
            if self.reject:
                raise ValueError("physical duplicate rejected")
            self.title = title
            return True

        def get_session_title(self, session_id, *, profile_name):
            return self.title

    primary = _DB("primary")
    physical = _DB("physical", reject=True)

    class _Candidates:
        def __enter__(self):
            # Read preference remains shared-primary first.
            return "coder", [primary, physical]

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        web_server,
        "_open_profile_session_candidates",
        lambda profile, *, read_only: _Candidates(),
    )
    monkeypatch.setattr(
        web_server,
        "_resolve_owned_session",
        lambda handles, profile_name, session_id: (
            primary,
            {"id": "duplicate"},
        ),
    )
    monkeypatch.setattr(
        web_server,
        "_owned_session_row",
        lambda db, session_id, profile_name: {"id": session_id},
    )

    with pytest.raises(HTTPException) as exc_info:
        web_server._rename_session(
            "duplicate",
            web_server.SessionRename(
                title="Changed",
                profile="coder",
            ),
        )

    assert exc_info.value.status_code == 400
    assert calls == ["physical"]
    assert physical.title == "Original"
    assert primary.title == "Original"
