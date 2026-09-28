"""One-shot observations for plugin-injected final replies.

Observers are process-local; a killed process cannot report an outcome. An absent
observation must therefore remain uncertain across restart.
"""

import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Callable, Optional


_observed_reply_turn = ContextVar("plugin_observed_reply_turn", default=None)


class DeliveryObserver:
    """Carry one callback across scheduling, queueing, and the final send."""

    def __init__(self, callback: Callable[[Optional[bool]], None]):
        self.callback = callback
        self.queued = False
        self.tool_started = False
        self.reply_ready = False
        self._completed = False
        self._lock = threading.Lock()

    def __call__(self, outcome: Optional[bool]) -> None:
        with self._lock:
            if self._completed:
                return
            self._completed = True
        try:
            self.callback(outcome)
        except (Exception, asyncio.CancelledError):
            # Plugin callbacks cannot change delivery or expose their payload in logs.
            pass


def delivery_observer(callback):
    if isinstance(callback, DeliveryObserver):
        return callback
    return DeliveryObserver(callback) if callable(callback) else None


def event_delivery_observer(event):
    observer = getattr(event, "_plugin_delivery_observer", None)
    return observer if isinstance(observer, DeliveryObserver) else None


@contextmanager
def observed_reply_turn(event):
    token = _observed_reply_turn.set(event_delivery_observer(event))
    try:
        yield
    finally:
        _observed_reply_turn.reset(token)


def is_observed_reply_turn() -> bool:
    return _observed_reply_turn.get() is not None


def current_delivery_observer():
    return _observed_reply_turn.get()


def final_reply_display(display):
    """Copy per-turn settings: observed replies use a single final text send."""
    config = dict(display.user_config)
    settings = dict(config.get("display") or {})
    platforms = dict(settings.get("platforms") or {})
    platform_settings = dict(platforms.get(display.platform_key) or {})
    platform_settings["streaming"] = False
    platforms[display.platform_key] = platform_settings
    settings["platforms"] = platforms
    config["display"] = settings
    return replace(
        display, user_config=config, tool_progress_enabled=False, progress_mode="off",
        _live_status_mode="off", log_mode_enabled=False, log_queue=None,
        interim_assistant_messages_enabled=False, _thinking_enabled=False,
        _native_slack_task_cards=False, needs_progress_queue=False,
        _display_surface_mode=lambda *_args, **_kwargs: "off",
    )


def observe_delivery(event, outcome: Optional[bool]) -> None:
    observer = event_delivery_observer(event)
    if observer is not None:
        observer(outcome)
