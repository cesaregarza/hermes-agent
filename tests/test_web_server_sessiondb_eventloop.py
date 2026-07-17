import ast
import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path

from hermes_cli import web_server


TARGET_HANDLERS = {
    "bulk_delete_sessions_endpoint",
    "count_empty_sessions_endpoint",
    "delete_empty_sessions_endpoint",
    "get_session_latest_descendant",
    "get_session_messages",
    "delete_session_endpoint",
    "export_session_endpoint",
    "prune_sessions_endpoint",
    "get_usage_analytics",
    "get_models_analytics",
}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_sessiondb_handlers_open_connections_inside_executor_helpers():
    tree = ast.parse(Path(web_server.__file__).read_text(encoding="utf-8"))
    handlers = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name in TARGET_HANDLERS
    }
    top_level_helpers = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert handlers.keys() == TARGET_HANDLERS

    for name, handler in handlers.items():
        helpers = {
            **top_level_helpers,
            **{
                node.name: node
                for node in handler.body
                if isinstance(node, ast.FunctionDef)
            },
        }
        offloaded = {
            arg.id
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and _call_name(node) == "to_thread"
            for arg in node.args[:1]
            if isinstance(arg, ast.Name)
        }

        def opens_session_db(
            helper_name: str,
            seen: set[str] | None = None,
        ) -> bool:
            """Follow local helper calls until a SessionDB opener is reached."""
            seen = set() if seen is None else seen
            if helper_name in seen:
                return False
            seen.add(helper_name)
            helper = helpers.get(helper_name)
            if helper is None:
                return False
            called = {
                call_name
                for node in ast.walk(helper)
                if isinstance(node, ast.Call)
                and (call_name := _call_name(node)) is not None
            }
            if called.intersection(
                {
                    "_open_session_db_for_profile",
                    "_open_profile_session_candidates",
                }
            ):
                return True
            return any(
                opens_session_db(call_name, seen)
                for call_name in called
                if call_name in helpers
            )

        db_open_owners = {
            helper_name for helper_name in offloaded
            if opens_session_db(helper_name)
        }
        assert db_open_owners, f"{name} does not offload SessionDB open + work"


def test_bulk_delete_sessiondb_work_runs_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def __init__(self):
            self.ids = {"one", "two"}

        def close(self):
            db_threads.append(threading.get_ident())

    @contextmanager
    def _open_candidates(profile=None, *, read_only=False):
        db_threads.append(threading.get_ident())
        db = _DB()
        try:
            yield "default", [db]
        finally:
            db.close()

    monkeypatch.setattr(
        web_server,
        "_open_profile_session_candidates",
        _open_candidates,
    )
    monkeypatch.setattr(
        web_server,
        "_all_owned_session_records",
        lambda handles, profile, *, deduplicate=True: [
            (handles[0], {"id": "one"}),
            (handles[0], {"id": "two"}),
        ],
    )

    def _delete_owned(db, ids, profile, **kwargs):
        db_threads.append(threading.get_ident())
        assert ids == ["one", "two"]
        assert profile == "default"
        db.ids.difference_update(ids)
        return 2

    monkeypatch.setattr(
        web_server,
        "_delete_owned_session_ids",
        _delete_owned,
    )
    monkeypatch.setattr(
        web_server,
        "_owned_session_row",
        lambda db, session_id, profile: (
            {"id": session_id}
            if session_id in db.ids
            else None
        ),
    )

    result = asyncio.run(
        web_server.bulk_delete_sessions_endpoint(
            web_server.BulkDeleteSessions(ids=["one", "two"])
        )
    )

    assert result == {"ok": True, "deleted": 2}
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)
