from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_multiplex_insights_captures_routed_profile_before_executor():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    event = MessageEvent(
        text="/insights 7",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
            profile="coder",
        ),
    )
    captured = {}

    class StubDB:
        def close(self):
            captured["closed"] = True

    class StubEngine:
        def __init__(self, db, profile_name=None):
            captured["db"] = db
            captured["profile_name"] = profile_name

        def generate(self, *, days, source):
            captured["days"] = days
            captured["source"] = source
            return {"ok": True}

        @staticmethod
        def format_gateway(report):
            assert report == {"ok": True}
            return "scoped insights"

    with (
        patch("hermes_state.SessionDB", StubDB),
        patch("agent.insights.InsightsEngine", StubEngine),
    ):
        result = await runner._handle_insights_command(event)

    assert result == "scoped insights"
    assert isinstance(captured.pop("db"), StubDB)
    assert captured == {
        "profile_name": "coder",
        "days": 7,
        "source": None,
        "closed": True,
    }
