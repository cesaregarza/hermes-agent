"""Profile-routing context carried through adapter sender authorization."""

from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner


def test_base_sender_auth_forwards_route_dimensions():
    seen = {}

    def check(user_id, chat_type, chat_id, **context):
        seen.update(
            user_id=user_id,
            chat_type=chat_type,
            chat_id=chat_id,
            **context,
        )
        return True

    adapter = SimpleNamespace(
        _authorization_check=check,
        name="test-adapter",
    )

    result = BasePlatformAdapter._is_sender_authorized(
        adapter,
        "user-1",
        "thread",
        "thread-1",
        scope_id="guild-1",
        thread_id="thread-1",
        parent_chat_id="channel-1",
    )

    assert result is True
    assert seen == {
        "user_id": "user-1",
        "chat_type": "thread",
        "chat_id": "thread-1",
        "scope_id": "guild-1",
        "thread_id": "thread-1",
        "parent_chat_id": "channel-1",
    }


def test_primary_auth_callback_resolves_guild_route_from_external_context():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.profile_routes = [
        ProfileRoute(
            name="guild-ops",
            platform="discord",
            profile="ops",
            guild_id="guild-1",
        )
    ]
    runner._primary_profile_name = "default"
    seen = []
    runner._is_user_authorized = lambda source: seen.append(source) or True

    callback = runner._make_adapter_auth_check(
        Platform.DISCORD,
        profile_name="default",
    )
    assert callback(
        "user-1",
        "thread",
        "thread-1",
        scope_id="guild-1",
        thread_id="thread-1",
        parent_chat_id="channel-1",
    ) is True

    assert len(seen) == 1
    source = seen[0]
    assert source.profile == "ops"
    assert source.transport_profile == "default"
    assert source.scope_id == "guild-1"
    assert source.thread_id == "thread-1"
    assert source.parent_chat_id == "channel-1"
