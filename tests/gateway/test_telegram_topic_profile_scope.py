"""Profile isolation for Telegram DM topic-mode persistence."""

import sqlite3
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


def _seed_profile_session(
    db: SessionDB,
    session_id: str,
    profile_name: str,
) -> None:
    namespace = "main" if profile_name == "default" else profile_name
    db.create_session(
        session_id,
        source="telegram",
        user_id="208214988",
        session_key=f"agent:{namespace}:telegram:dm:208214988:17585",
        profile_name=profile_name,
    )


def test_telegram_topic_state_isolated_by_runtime_profile(tmp_path):
    db = SessionDB(tmp_path / "state.db", profile_name="default")
    _seed_profile_session(db, "default-session", "default")
    _seed_profile_session(db, "coder-session", "coder")

    for profile_name, session_id in (
        ("default", "default-session"),
        ("coder", "coder-session"),
    ):
        namespace = "main" if profile_name == "default" else profile_name
        db.enable_telegram_topic_mode(
            chat_id="208214988",
            user_id="208214988",
            profile_name=profile_name,
        )
        db.bind_telegram_topic(
            chat_id="208214988",
            thread_id="17585",
            user_id="208214988",
            session_key=f"agent:{namespace}:telegram:dm:208214988:17585",
            session_id=session_id,
            profile_name=profile_name,
        )

    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="default",
    )["session_id"] == "default-session"
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="coder",
    )["session_id"] == "coder-session"
    assert db.get_telegram_topic_binding_by_session(
        session_id="coder-session",
        profile_name="default",
    ) is None
    assert db.is_telegram_session_linked_to_topic(
        session_id="coder-session",
        profile_name="coder",
    ) is True

    db.disable_telegram_topic_mode(
        chat_id="208214988",
        profile_name="default",
    )

    assert db.is_telegram_topic_mode_enabled(
        chat_id="208214988",
        user_id="208214988",
        profile_name="default",
    ) is False
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="default",
    ) is None
    assert db.is_telegram_topic_mode_enabled(
        chat_id="208214988",
        user_id="208214988",
        profile_name="coder",
    ) is True
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="coder",
    )["session_id"] == "coder-session"


def test_unlinked_telegram_sessions_are_profile_scoped(tmp_path):
    db = SessionDB(tmp_path / "state.db", profile_name="default")
    _seed_profile_session(db, "default-session", "default")
    _seed_profile_session(db, "coder-session", "coder")

    default_rows = db.list_unlinked_telegram_sessions_for_user(
        chat_id="208214988",
        user_id="208214988",
        profile_name="default",
    )
    coder_rows = db.list_unlinked_telegram_sessions_for_user(
        chat_id="208214988",
        user_id="208214988",
        profile_name="coder",
    )

    assert [row["id"] for row in default_rows] == ["default-session"]
    assert [row["id"] for row in coder_rows] == ["coder-session"]


def test_v2_topic_rows_migrate_to_persisted_database_owner(tmp_path):
    db = SessionDB(tmp_path / "state.db", profile_name="research")
    _seed_profile_session(db, "legacy-session", "research")
    with db._lock:
        db._conn.executescript(
            """
            DROP TABLE IF EXISTS telegram_dm_topic_bindings;
            DROP TABLE IF EXISTS telegram_dm_topic_mode;
            CREATE TABLE telegram_dm_topic_mode (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                activated_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                has_topics_enabled INTEGER,
                allows_users_to_create_topics INTEGER,
                capability_checked_at REAL,
                intro_message_id TEXT,
                pinned_message_id TEXT
            );
            CREATE TABLE telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            );
            INSERT INTO telegram_dm_topic_mode (
                chat_id, user_id, enabled, activated_at, updated_at
            ) VALUES ('208214988', '208214988', 1, 1.0, 1.0);
            INSERT INTO telegram_dm_topic_bindings (
                chat_id, thread_id, user_id, session_key, session_id,
                managed_mode, linked_at, updated_at
            ) VALUES (
                '208214988', '17585', '208214988',
                'agent:main:telegram:dm:208214988:17585',
                'legacy-session', 'auto', 1.0, 1.0
            );
            INSERT INTO state_meta (key, value)
            VALUES ('telegram_dm_topic_schema_version', '2')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value;
            """
        )

    db.apply_telegram_topic_migration()

    assert db.get_meta("telegram_dm_topic_schema_version") == "3"
    binding = db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="research",
    )
    assert binding is not None
    assert binding["profile_name"] == "research"
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="default",
    ) is None


def test_v3_migration_failure_rolls_back_and_retry_preserves_legacy_data(tmp_path):
    db = SessionDB(tmp_path / "state.db", profile_name="research")
    _seed_profile_session(db, "valid-session", "research")
    with db._lock:
        db._conn.execute("PRAGMA foreign_keys = OFF")
        db._conn.executescript(
            """
            CREATE TABLE telegram_dm_topic_mode (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                activated_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                has_topics_enabled INTEGER,
                allows_users_to_create_topics INTEGER,
                capability_checked_at REAL,
                intro_message_id TEXT,
                pinned_message_id TEXT
            );
            CREATE TABLE telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL
                    REFERENCES sessions(id) ON DELETE CASCADE,
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            );
            INSERT INTO telegram_dm_topic_mode (
                chat_id, user_id, enabled, activated_at, updated_at
            ) VALUES ('208214988', '208214988', 1, 1.0, 1.0);
            INSERT INTO telegram_dm_topic_bindings (
                chat_id, thread_id, user_id, session_key, session_id,
                managed_mode, linked_at, updated_at
            ) VALUES
                (
                    '208214988', '17585', '208214988',
                    'agent:research:telegram:dm:208214988:17585',
                    'valid-session', 'auto', 1.0, 1.0
                ),
                (
                    '208214988', '99999', '208214988',
                    'agent:research:telegram:dm:208214988:99999',
                    'orphan-session', 'auto', 1.0, 1.0
                );
            """
        )
        db._conn.execute("PRAGMA foreign_keys = ON")

    with pytest.raises(sqlite3.IntegrityError):
        db.apply_telegram_topic_migration()

    mode_columns = {
        row[1]
        for row in db._conn.execute(
            "PRAGMA table_info('telegram_dm_topic_mode')"
        )
    }
    binding_columns = {
        row[1]
        for row in db._conn.execute(
            "PRAGMA table_info('telegram_dm_topic_bindings')"
        )
    }
    assert "profile_name" not in mode_columns
    assert "profile_name" not in binding_columns
    assert db._conn.execute(
        "SELECT COUNT(*) FROM telegram_dm_topic_bindings"
    ).fetchone()[0] == 2
    assert not db._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name IN ("
        "'telegram_dm_topic_mode_new', "
        "'telegram_dm_topic_bindings_new'"
        ")"
    ).fetchall()

    db._conn.execute(
        "DELETE FROM telegram_dm_topic_bindings WHERE session_id = ?",
        ("orphan-session",),
    )
    db.apply_telegram_topic_migration()

    binding = db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="research",
    )
    assert binding is not None
    assert binding["session_id"] == "valid-session"
    assert db.get_meta("telegram_dm_topic_schema_version") == "3"


def test_shared_adapter_prunes_routed_profile_not_credential_owner(tmp_path):
    db = SessionDB(tmp_path / "state.db", profile_name="default")
    _seed_profile_session(db, "default-session", "default")
    _seed_profile_session(db, "coder-session", "coder")
    for profile_name, session_id in (
        ("default", "default-session"),
        ("coder", "coder-session"),
    ):
        namespace = "main" if profile_name == "default" else profile_name
        db.enable_telegram_topic_mode(
            chat_id="208214988",
            user_id="208214988",
            profile_name=profile_name,
        )
        db.bind_telegram_topic(
            chat_id="208214988",
            thread_id="17585",
            user_id="208214988",
            session_key=f"agent:{namespace}:telegram:dm:208214988:17585",
            session_id=session_id,
            profile_name=profile_name,
        )

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._session_store = SimpleNamespace(_db=db)
    adapter._profile_name = "default"
    adapter._profile_routes_enabled = True

    adapter._prune_stale_dm_topic_binding(
        "208214988",
        "17585",
        profile_name="coder",
    )

    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="coder",
    ) is None
    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="default",
    )["session_id"] == "default-session"
    assert db.is_telegram_topic_mode_enabled(
        chat_id="208214988",
        user_id="208214988",
        profile_name="default",
    ) is True


def test_shared_adapter_without_runtime_profile_skips_destructive_prune(tmp_path):
    db = SessionDB(tmp_path / "state.db", profile_name="default")
    _seed_profile_session(db, "default-session", "default")
    db.enable_telegram_topic_mode(
        chat_id="208214988",
        user_id="208214988",
        profile_name="default",
    )
    db.bind_telegram_topic(
        chat_id="208214988",
        thread_id="17585",
        user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:17585",
        session_id="default-session",
        profile_name="default",
    )

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._session_store = SimpleNamespace(_db=db)
    adapter._profile_name = "default"
    adapter._profile_routes_enabled = True

    adapter._prune_stale_dm_topic_binding("208214988", "17585")
    adapter._profile_name = None
    adapter._prune_stale_dm_topic_binding("208214988", "17585")
    adapter._prune_stale_dm_topic_binding(
        "208214988",
        "17585",
        profile_name="../coder",
    )

    assert db.get_telegram_topic_binding(
        chat_id="208214988",
        thread_id="17585",
        profile_name="default",
    ) is not None


def test_telegram_send_metadata_carries_routed_runtime_profile():
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id="17585",
        profile="coder",
        transport_profile="default",
    )

    metadata = runner._thread_metadata_for_source(source)

    assert metadata["runtime_profile"] == "coder"


@pytest.mark.parametrize(
    "store_name",
    ["_model_picker_state", "_choice_picker_state"],
)
def test_shared_adapter_picker_state_isolated_by_prompt_message(store_name):
    adapter = object.__new__(TelegramAdapter)
    store = {}
    setattr(adapter, store_name, store)
    default_state = {"msg_id": 101, "profile": "default"}
    coder_state = {"msg_id": 202, "profile": "coder"}
    adapter._remember_picker_state(store, "208214988", 101, default_state)
    adapter._remember_picker_state(store, "208214988", 202, coder_state)

    default_key, resolved_default = adapter._picker_state_for_query(
        store,
        SimpleNamespace(message=SimpleNamespace(message_id=101)),
        "208214988",
    )
    coder_key, resolved_coder = adapter._picker_state_for_query(
        store,
        SimpleNamespace(message=SimpleNamespace(message_id=202)),
        "208214988",
    )
    unknown_key, resolved_unknown = adapter._picker_state_for_query(
        store,
        SimpleNamespace(message=SimpleNamespace(message_id=303)),
        "208214988",
    )

    assert default_key == ("208214988", "101")
    assert coder_key == ("208214988", "202")
    assert unknown_key == ("208214988", "303")
    assert resolved_default is default_state
    assert resolved_coder is coder_state
    assert resolved_unknown is None

    adapter._drop_picker_state(
        store,
        "208214988",
        default_key,
        default_state,
    )

    assert ("208214988", "101") not in store
    assert store["208214988"] is coder_state
    assert store[("208214988", "202")] is coder_state


def test_telegram_cooldowns_are_isolated_by_runtime_profile():
    runner = object.__new__(GatewayRunner)
    default_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        profile="default",
    )
    coder_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        profile="coder",
    )

    assert runner._should_send_telegram_lobby_reminder(default_source) is True
    assert runner._should_send_telegram_lobby_reminder(default_source) is False
    assert runner._should_send_telegram_lobby_reminder(coder_source) is True

    assert runner._should_send_telegram_capability_hint(default_source) is True
    assert runner._should_send_telegram_capability_hint(default_source) is False
    assert runner._should_send_telegram_capability_hint(coder_source) is True
