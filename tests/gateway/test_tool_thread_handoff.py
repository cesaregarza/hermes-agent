"""Hosted contracts for the real MCP pre-transport thread handoff."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway import tool_thread_handoff as handoff
from tools import mcp_tool_config, mcp_tool_handlers, mcp_tool_session_metadata

PREFIX = 'com.nousresearch.hermes/'


class Runner:
    pass


def setup_handoff(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    source = SessionSource(platform=Platform.DISCORD, chat_id='100', chat_type='group',
                           user_id='200', guild_id='300', message_id='400')
    entry = SimpleNamespace(origin=source, session_id='session', session_key='key')
    message = SimpleNamespace(id=401, author=SimpleNamespace(id=200, bot=False),
                              channel=SimpleNamespace(id=100), guild=SimpleNamespace(id=300),
                              content='Prepare the ingredient shopping list')
    thread = SimpleNamespace(id=401, parent_id=100, name='Ingredients')
    message.create_thread = AsyncMock(return_value=thread)
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter = SimpleNamespace(
        _client=SimpleNamespace(user=SimpleNamespace(id=500), get_channel=Mock(return_value=channel)),
        _derive_auto_thread_name=lambda content: 'Ingredients', _threads=SimpleNamespace(mark=Mock()),
        _dedup=SimpleNamespace(is_duplicate=Mock()), _message_handler=object(), handle_message=AsyncMock(),
    )
    runner = Runner()
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running, runner._draining = True, False
    runner.async_session_store = SimpleNamespace(lookup_by_session_key=AsyncMock(return_value=entry))
    runner._is_user_authorized = Mock(return_value=True)
    runner._adapter_for_source = Mock(return_value=adapter)
    meta = {PREFIX + k: v for k, v in {'platform': 'discord', 'session_key': 'key',
            'session_id': 'session', 'chat_id': '100', 'user_id': '200', 'message_id': '401', 'thread_id': ''}.items()}
    monkeypatch.setattr(mcp_tool_config, '_load_mcp_config', lambda: {'jobs': {
        'gateway_thread_handoff': {'tools': ['submit'], 'channels': ['100']}}})
    monkeypatch.setattr(mcp_tool_session_metadata, 'build_session_context_meta', lambda server: meta)
    monkeypatch.setattr(mcp_tool_handlers, '_trust_gate_check', lambda *args: None)
    monkeypatch.setattr(mcp_tool_handlers, '_check_circuit_breaker', lambda *args: None)
    transport = Mock(side_effect=AssertionError('parent must not reach MCP transport'))
    monkeypatch.setattr(mcp_tool_handlers, '_acquire_call_server', transport)
    handoff.install_router(runner)
    return runner, entry, adapter, message, meta, transport


@pytest.mark.asyncio
async def test_real_mcp_handler_hands_off_once_before_transport(tmp_path, monkeypatch):
    runner, entry, adapter, message, meta, transport = setup_handoff(tmp_path, monkeypatch)
    handler = mcp_tool_handlers._make_tool_handler(
        'jobs', 'submit', 10, gateway_thread_handoff={'tools': ['submit'], 'channels': ['100']})
    args = {'task': 'ingredients for this week', 'requested_capability': 'shopping'}
    try:
        first = await asyncio.to_thread(handler, args)
        repeated = await asyncio.to_thread(handler, args)
        assert first == repeated and 'moved_to_thread' in first
        message.create_thread.assert_awaited_once()
        adapter.handle_message.assert_awaited_once()
        transport.assert_not_called()
        event = adapter.handle_message.call_args.args[0]
        assert event.source.user_id == '200' and event.source.parent_chat_id == '100'
        assert event.source.thread_id == event.source.chat_id == '401'
        assert event.message_id == event.source.message_id == '401'
        assert event.internal and not event.allow_gateway_control
        assert entry.origin.thread_id is None and entry.origin.message_id == '400'
        assert 'ingredients for this week' in event.text
        # Ordinary chat does not call the configured job tool. Existing thread
        # calls execute normally; the adapter has already bound their source.
        setting = handoff.resolve_handoff_config({'tools': ['submit'], 'channels': ['100']})
        assert handoff.route_tool_call('jobs', 'read', {}, meta, setting) is None
        assert handoff.route_tool_call('jobs', 'submit', args, {**meta, PREFIX+'thread_id': '401'}, setting) is None
    finally:
        handoff.clear_router(runner)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['wrong_user', 'creation_timeout', 'session_reset', 'wrong_author', 'wrong_channel', 'wrong_guild', 'bot_author'])
async def test_failed_or_uncertain_handoff_never_submits_or_replays(tmp_path, monkeypatch, failure):
    runner, entry, adapter, message, meta, transport = setup_handoff(tmp_path, monkeypatch)
    if failure == 'wrong_user':
        meta[PREFIX+'user_id'] = '999'
    elif failure == 'creation_timeout':
        message.create_thread.side_effect = TimeoutError('ambiguous response')
    elif failure == 'wrong_author':
        message.author.id = 999
    elif failure == 'wrong_channel':
        message.channel.id = 999
    elif failure == 'wrong_guild':
        message.guild.id = 999
    elif failure == 'bot_author':
        message.author.bot = True
    else:
        async def reset_during_create(**kwargs):
            entry.session_id = 'replacement'
            return SimpleNamespace(id=401, parent_id=100, name='Ingredients')
        message.create_thread.side_effect = reset_during_create
    handler = mcp_tool_handlers._make_tool_handler(
        'jobs', 'submit', 10, gateway_thread_handoff={'tools': ['submit'], 'channels': ['100']})
    try:
        first = await asyncio.to_thread(handler, {'task': 'bounded job'})
        second = await asyncio.to_thread(handler, {'task': 'bounded job'})
        assert 'moved_to_thread' not in first + second
        assert message.create_thread.await_count <= 1
        adapter.handle_message.assert_not_called()
        transport.assert_not_called()
    finally:
        handoff.clear_router(runner)


@pytest.mark.asyncio
@pytest.mark.parametrize('registration', ['live', 'cached'])
async def test_registered_handoff_uses_frozen_config_without_reload(tmp_path, monkeypatch, registration):
    from tools import mcp_tool, mcp_tool_registration, registry

    runner, entry, adapter, message, meta, transport = setup_handoff(tmp_path, monkeypatch)
    config = {'gateway_thread_handoff': {'tools': ['submit'], 'channels': [100]}}
    tool = SimpleNamespace(name='submit', description='Submit a job', inputSchema={}, annotations=None)
    private_registry = registry.ToolRegistry()
    monkeypatch.setattr(registry, 'registry', private_registry)
    monkeypatch.setattr(mcp_tool_registration, '_write_schema_cache', Mock())
    monkeypatch.setattr(mcp_tool_registration, '_select_utility_schemas', lambda *args: [])
    for attr in ('_lazy_server_configs', '_lazy_server_fingerprints', '_lazy_server_tool_names'):
        monkeypatch.setattr(mcp_tool, attr, {})
    # Registration already has config, so even handler construction must not
    # reload dotenv/discover plugins. Both live and cached registrations carry it.
    loader = Mock(side_effect=AssertionError('MCP config reload in handoff'))
    monkeypatch.setattr(mcp_tool_config, '_load_mcp_config', loader)
    if registration == 'live':
        names = mcp_tool_registration._register_server_tools(
            'jobs', SimpleNamespace(_tools=[tool], tool_timeout=10), config)
    else:
        names = mcp_tool_registration._register_from_cache_sync(
            'jobs', config, {'tools': [vars(tool)], 'utility_tools': []})
    assert len(names) == 1
    handler = private_registry.get_entry(names[0]).handler
    config['gateway_thread_handoff']['tools'].clear()
    config['gateway_thread_handoff']['channels'].clear()
    try:
        first = await asyncio.to_thread(handler, {'task': 'shopping'})
        replay = await asyncio.to_thread(handler, {'task': 'shopping'})
        collision = await asyncio.to_thread(handler, {'task': 'different job'})
        assert first == replay and 'moved_to_thread' in first
        assert 'moved_to_thread' not in collision
        adapter.handle_message.assert_awaited_once()
        message.create_thread.assert_awaited_once()
        transport.assert_not_called()
        loader.assert_not_called()
    finally:
        handoff.clear_router(runner)


def test_unconfigured_handler_keeps_transport_path_without_config_reload(monkeypatch):
    loader = Mock(side_effect=AssertionError('configuration must not reload'))
    monkeypatch.setattr(mcp_tool_config, '_load_mcp_config', loader)
    monkeypatch.setattr(mcp_tool_session_metadata, 'build_session_context_meta', lambda server: None)
    monkeypatch.setattr(mcp_tool_handlers, '_trust_gate_check', lambda *args: None)
    monkeypatch.setattr(mcp_tool_handlers, '_check_circuit_breaker', lambda *args: None)
    transport = Mock(return_value=(None, 'transport reached'))
    monkeypatch.setattr(mcp_tool_handlers, '_acquire_call_server', transport)
    handler = mcp_tool_handlers._make_tool_handler('ordinary', 'read', 10)
    assert handler({}) == handler({}) == 'transport reached'
    assert transport.call_count == 2
    loader.assert_not_called()


@pytest.mark.parametrize('setting', [{}, {'tools': ['submit'], 'channels': [True]},
                                     {'tools': [None], 'channels': ['100']},
                                     {'tools': ['submit'], 'channels': ['invalid']}])
def test_invalid_handoff_config_fails_closed(setting):
    resolved = handoff.resolve_handoff_config(setting)
    result = handoff.route_tool_call('jobs', 'submit', {}, None, resolved)
    assert 'Invalid gateway thread handoff configuration' in result
