"""Tests for Telegram text message aggregation.

When a user sends a long message, Telegram clients split it into multiple
updates.  The TelegramAdapter should buffer rapid successive text messages
from the same session and aggregate them before dispatching.
"""

import asyncio
import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    _message_event_sender_identity,
)


def _make_adapter():
    """Create a minimal TelegramAdapter for testing text batching."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter.config = config
    adapter._running = True
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._drop_delayed_deliveries = False
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_text_batch_senders = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._pending_photo_batch_senders = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._media_group_senders = {}
    adapter._media_group_ids = {}
    adapter._polling_error_task = None
    adapter._polling_heartbeat_task = None
    adapter._app = None
    adapter._bot = None
    adapter._set_status_indicator = AsyncMock()
    adapter._release_platform_lock = lambda: None
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(
    text: str,
    chat_id: str = "12345",
    *,
    chat_type: str = "dm",
    user_id: str | None = None,
    thread_id: str | None = None,
    message_id: str | None = None,
    profile: str | None = None,
    message_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=message_type,
        message_id=message_id,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            thread_id=thread_id,
            profile=profile,
        ),
    )


class TestTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_adapter()
        event = _make_event("hello world")

        await adapter._enqueue_text_event(event)

        # Not dispatched yet
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        """Two rapid messages from the same chat should be merged."""
        adapter = _make_adapter()

        await adapter._enqueue_text_event(_make_event("This is part one of a long"))
        await asyncio.sleep(0.02)  # small gap, within batch window
        await adapter._enqueue_text_event(_make_event("message that was split by Telegram."))

        # Not dispatched yet (timer restarted)
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "part one" in dispatched.text
        assert "split by Telegram" in dispatched.text

    @pytest.mark.asyncio
    async def test_three_way_split_aggregated(self):
        """Three rapid messages should all merge."""
        adapter = _make_adapter()

        await adapter._enqueue_text_event(_make_event("chunk 1"))
        await asyncio.sleep(0.02)
        await adapter._enqueue_text_event(_make_event("chunk 2"))
        await asyncio.sleep(0.02)
        await adapter._enqueue_text_event(_make_event("chunk 3"))

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "chunk 1" in text
        assert "chunk 2" in text
        assert "chunk 3" in text

    @pytest.mark.asyncio
    async def test_different_chats_not_merged(self):
        """Messages from different chats should be separate batches."""
        adapter = _make_adapter()

        await adapter._enqueue_text_event(_make_event("from user A", chat_id="111"))
        await adapter._enqueue_text_event(_make_event("from user B", chat_id="222"))

        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_shared_thread_text_batches_do_not_cross_senders(self):
        """A shared session must not attach Bob's text to Alice's event."""
        adapter = _make_adapter()
        adapter.config.extra["group_sessions_per_user"] = False
        alice = _make_event(
            "from Alice",
            chat_type="group",
            user_id="alice",
            thread_id="topic-7",
            message_id="msg-a",
        )
        bob = _make_event(
            "from Bob",
            chat_type="group",
            user_id="bob",
            thread_id="topic-7",
            message_id="msg-b",
        )

        await adapter._enqueue_text_event(alice)
        await adapter._enqueue_text_event(bob)
        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2
        dispatched = {
            call.args[0].source.user_id: (call.args[0].text, call.args[0].message_id)
            for call in adapter.handle_message.call_args_list
        }
        assert dispatched == {
            "alice": ("from Alice", "msg-a"),
            "bob": ("from Bob", "msg-b"),
        }

    @pytest.mark.asyncio
    async def test_observe_mode_uses_private_pre_redaction_sender_identity(self):
        adapter = _make_adapter()
        adapter.config.extra["group_sessions_per_user"] = False

        def observed_event(text: str, user_id: str) -> tuple[MessageEvent, tuple[str, ...]]:
            attributed = _make_event(
                text,
                chat_id="observed-group",
                chat_type="group",
                user_id=user_id,
                thread_id="topic-7",
            )
            identity = _message_event_sender_identity(attributed)
            assert identity is not None
            public = dataclasses.replace(
                attributed,
                text=f"[{user_id}]\n{text}",
                source=dataclasses.replace(
                    attributed.source,
                    user_id=None,
                    user_name=None,
                    user_id_alt=None,
                ),
            )
            return public, identity

        alice_one, alice_identity = observed_event("A1", "alice")
        alice_two, alice_identity_again = observed_event("A2", "alice")
        bob, bob_identity = observed_event("B", "bob")

        await adapter._enqueue_text_event(
            alice_one, batch_sender_identity=alice_identity,
        )
        await adapter._enqueue_text_event(
            alice_two, batch_sender_identity=alice_identity_again,
        )
        await adapter._enqueue_text_event(
            bob, batch_sender_identity=bob_identity,
        )
        await asyncio.sleep(0.2)

        dispatched = [call.args[0] for call in adapter.handle_message.call_args_list]
        assert [event.text for event in dispatched] == ["[alice]\nA1\n[alice]\nA2", "[bob]\nB"]
        assert all(event.source.user_id is None for event in dispatched)
        assert all("batch_sender_identity" not in event.metadata for event in dispatched)
        assert adapter._pending_text_batch_senders == {}

    @pytest.mark.asyncio
    async def test_batch_cleans_up_after_flush(self):
        """After flushing, internal state should be clean."""
        adapter = _make_adapter()

        await adapter._enqueue_text_event(_make_event("test"))
        await asyncio.sleep(0.2)

        assert len(adapter._pending_text_batches) == 0
        assert len(adapter._pending_text_batch_tasks) == 0

    @pytest.mark.asyncio
    async def test_dm_topic_batching_recovers_thread_before_keying(self):
        """DM-topic text batches should use the recovered topic lane."""
        adapter = _make_adapter()
        adapter.set_topic_recovery_fn(
            lambda source: "222" if str(source.thread_id or "") == "1" else None
        )
        event = MessageEvent(
            text="hello from DM topic",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
                user_id="user-1",
                thread_id="1",
            ),
        )

        await adapter._enqueue_text_event(event)
        assert event.source.thread_id == "222"
        assert list(adapter._pending_text_batches) == [adapter._text_batch_key(event)]

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.source.thread_id == "222"

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_text_batch_without_dispatch(self):
        """Disconnect should not let buffered text flush into a stale run."""
        adapter = _make_adapter()

        await adapter._enqueue_text_event(_make_event("stale text"))
        await adapter.disconnect()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_pending_text_flush_before_dispatch(self):
        """A pending text flush should drop its event if teardown wins the race."""
        adapter = _make_adapter()

        await adapter._enqueue_text_event(_make_event("stale text"))
        adapter._mark_disconnected()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_late_text_batch_enqueue(self):
        """Late update handlers should not schedule batches after teardown starts."""
        adapter = _make_adapter()
        adapter._mark_disconnected()

        await adapter._enqueue_text_event(_make_event("late text"))
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_pending_photo_flush_before_dispatch(self):
        """A pending photo batch should not dispatch after disconnect starts."""
        adapter = _make_adapter()
        adapter._media_batch_delay_seconds = 0.1
        event = _make_event("photo caption")
        event.media_urls = ["/tmp/photo.jpg"]
        event.media_types = ["image/jpeg"]

        await adapter._enqueue_photo_event("chat:photo-burst", event)
        adapter._mark_disconnected()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_photo_batches == {}
        assert adapter._pending_photo_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_shared_thread_photo_bursts_do_not_cross_senders(self):
        adapter = _make_adapter()
        adapter.config.extra["group_sessions_per_user"] = False
        adapter._media_batch_delay_seconds = 0.05
        message = SimpleNamespace(media_group_id=None)
        alice = _make_event(
            "Alice photo",
            chat_type="group",
            user_id="alice",
            thread_id="topic-7",
            message_id="photo-a",
            message_type=MessageType.PHOTO,
        )
        bob = _make_event(
            "Bob photo",
            chat_type="group",
            user_id="bob",
            thread_id="topic-7",
            message_id="photo-b",
            message_type=MessageType.PHOTO,
        )
        alice.media_urls = ["/tmp/alice.jpg"]
        alice.media_types = ["image/jpeg"]
        bob.media_urls = ["/tmp/bob.jpg"]
        bob.media_types = ["image/jpeg"]

        alice_key = adapter._photo_batch_key(alice, message)
        bob_key = adapter._photo_batch_key(bob, message)
        assert alice_key == bob_key
        await adapter._enqueue_photo_event(alice_key, alice)
        await adapter._enqueue_photo_event(bob_key, bob)
        await asyncio.sleep(0.15)

        assert adapter.handle_message.call_count == 2
        dispatched = {
            call.args[0].source.user_id: (
                call.args[0].media_urls,
                call.args[0].message_id,
            )
            for call in adapter.handle_message.call_args_list
        }
        assert dispatched == {
            "alice": (["/tmp/alice.jpg"], "photo-a"),
            "bob": (["/tmp/bob.jpg"], "photo-b"),
        }

    def test_photo_and_album_keys_include_profile_and_chat(self):
        adapter = _make_adapter()
        adapter.config.extra["group_sessions_per_user"] = False
        profile_a = _make_event(
            "photo",
            chat_type="group",
            user_id="alice",
            profile="profile-a",
            message_type=MessageType.PHOTO,
        )
        profile_b = _make_event(
            "photo",
            chat_type="group",
            user_id="alice",
            profile="profile-b",
            message_type=MessageType.PHOTO,
        )
        other_chat = _make_event(
            "photo",
            chat_id="67890",
            chat_type="group",
            user_id="alice",
            profile="profile-a",
            message_type=MessageType.PHOTO,
        )

        plain_photo = SimpleNamespace(media_group_id=None)
        assert adapter._photo_batch_key(profile_a, plain_photo) != adapter._photo_batch_key(
            profile_b, plain_photo,
        )
        assert adapter._media_group_batch_key(
            profile_a, "album-reused",
        ) != adapter._media_group_batch_key(profile_b, "album-reused")
        assert adapter._media_group_batch_key(
            profile_a, "album-reused",
        ) != adapter._media_group_batch_key(other_chat, "album-reused")

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_pending_media_group_flush_before_dispatch(self):
        """A pending media group should not dispatch after disconnect starts."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = _make_adapter()
        event = _make_event("album caption")
        event.media_urls = ["/tmp/photo.jpg"]
        event.media_types = ["image/jpeg"]

        with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 0.1):
            await adapter._queue_media_group_event("album-1", event)
            adapter._mark_disconnected()
            await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._media_group_events == {}
        assert adapter._media_group_tasks == {}

    @pytest.mark.asyncio
    async def test_stale_media_group_flush_does_not_clear_newer_task(self):
        """A cancelled album flush must not erase the replacement task handle."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = _make_adapter()
        first = _make_event("first album caption")
        first.media_urls = ["/tmp/first.jpg"]
        first.media_types = ["image/jpeg"]
        second = _make_event("second album caption")
        second.media_urls = ["/tmp/second.jpg"]
        second.media_types = ["image/jpeg"]
        batch_key = adapter._media_group_batch_key(first, "album-race")

        with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 1.0):
            await adapter._queue_media_group_event("album-race", first)
            first_task = adapter._media_group_tasks[batch_key]
            await asyncio.sleep(0)

            await adapter._queue_media_group_event("album-race", second)
            replacement_task = adapter._media_group_tasks[batch_key]
            assert replacement_task is not first_task

            await asyncio.sleep(0)
            assert adapter._media_group_tasks.get(batch_key) is replacement_task

            replacement_task.cancel()
            await asyncio.gather(replacement_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_shared_album_id_does_not_cross_senders(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = _make_adapter()
        adapter.config.extra["group_sessions_per_user"] = False
        alice = _make_event(
            "Alice album",
            chat_type="group",
            user_id="alice",
            thread_id="topic-7",
            message_id="album-a",
            message_type=MessageType.PHOTO,
        )
        bob = _make_event(
            "Bob album",
            chat_type="group",
            user_id="bob",
            thread_id="topic-7",
            message_id="album-b",
            message_type=MessageType.PHOTO,
        )
        alice.media_urls = ["/tmp/alice.jpg"]
        alice.media_types = ["image/jpeg"]
        bob.media_urls = ["/tmp/bob.jpg"]
        bob.media_types = ["image/jpeg"]

        with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 0.05):
            await adapter._queue_media_group_event("album-shared", alice)
            await adapter._queue_media_group_event("album-shared", bob)
            await asyncio.sleep(0.15)

        assert adapter.handle_message.call_count == 2
        dispatched = {
            call.args[0].source.user_id: (
                call.args[0].media_urls,
                call.args[0].message_id,
            )
            for call in adapter.handle_message.call_args_list
        }
        assert dispatched == {
            "alice": (["/tmp/alice.jpg"], "album-a"),
            "bob": (["/tmp/bob.jpg"], "album-b"),
        }

    @pytest.mark.asyncio
    async def test_cancel_pending_delivery_tasks_skips_current_polling_error_task(self):
        """The teardown helper must not cancel the coroutine doing cleanup."""
        adapter = _make_adapter()
        current_task = asyncio.current_task()
        stale_task = asyncio.create_task(asyncio.sleep(60))
        adapter._pending_text_batches["text"] = _make_event("text")
        adapter._pending_text_batch_tasks["text"] = stale_task
        adapter._polling_error_task = current_task

        await adapter._cancel_pending_delivery_tasks()

        assert stale_task.done()
        assert stale_task.cancelled()
        assert not current_task.cancelled()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}
        assert adapter._polling_error_task is current_task

    @pytest.mark.asyncio
    async def test_disconnect_cancels_all_pending_delivery_task_maps(self):
        """Photo/media/polling delayed tasks are awaited and queues are cleared."""
        adapter = _make_adapter()
        tasks = [asyncio.create_task(asyncio.sleep(60)) for _ in range(4)]
        adapter._pending_text_batches["text"] = _make_event("text")
        adapter._pending_text_batch_tasks["text"] = tasks[0]
        adapter._pending_photo_batches["photo"] = _make_event("photo")
        adapter._pending_photo_batch_tasks["photo"] = tasks[1]
        adapter._media_group_events["media"] = _make_event("media")
        adapter._media_group_tasks["media"] = tasks[2]
        adapter._polling_error_task = tasks[3]

        await adapter.disconnect()

        assert all(task.done() for task in tasks)
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}
        assert adapter._pending_photo_batches == {}
        assert adapter._pending_photo_batch_tasks == {}
        assert adapter._media_group_events == {}
        assert adapter._media_group_tasks == {}
        assert adapter._polling_error_task is None
