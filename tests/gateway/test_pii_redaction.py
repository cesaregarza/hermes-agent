"""Tests for PII redaction in gateway session context prompts."""

from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context_prompt,
    _hash_id,
    _hash_message_id,
    _hash_sender_id,
    _hash_chat_id,
    _hash_thread_id,
)
from gateway.config import Platform, HomeChannel


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestHashHelpers:
    def test_hash_id_deterministic(self):
        assert _hash_id("12345") == _hash_id("12345")

    def test_hash_id_12_hex_chars(self):
        h = _hash_id("user-abc")
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_sender_id_prefix(self):
        assert _hash_sender_id("12345").startswith("user_")
        assert len(_hash_sender_id("12345")) == 17  # "user_" + 12

    def test_hash_chat_id_preserves_prefix(self):
        result = _hash_chat_id("telegram:12345")
        assert result.startswith("telegram:")
        assert "12345" not in result

    def test_hash_chat_id_no_prefix(self):
        result = _hash_chat_id("12345")
        assert len(result) == 12
        assert "12345" not in result

    def test_hash_thread_id_hashes_entire_identifier(self):
        result = _hash_thread_id("+15551234567:topic")
        assert result == f"thread_{_hash_id('+15551234567:topic')}"
        assert "+15551234567" not in result

    def test_hash_message_id_uses_message_namespace(self):
        result = _hash_message_id("wamid.HBgLMTU1NTEyMzQ1NjcVAgASGBQ")
        assert result.startswith("message_")


# ---------------------------------------------------------------------------
# Integration: build_session_context_prompt
# ---------------------------------------------------------------------------

def _make_context(
    user_id="user-123",
    user_name=None,
    chat_id="telegram:99999",
    chat_name=None,
    chat_type="dm",
    platform=Platform.TELEGRAM,
    home_channels=None,
):
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
    )
    return SessionContext(
        source=source,
        connected_platforms=[platform],
        home_channels=home_channels or {},
    )


class TestBuildSessionContextPromptRedaction:
    def test_no_redaction_by_default(self):
        ctx = _make_context(user_id="user-123")
        prompt = build_session_context_prompt(ctx)
        assert "user-123" in prompt

    def test_user_id_hashed_when_redact_pii(self):
        ctx = _make_context(user_id="user-123")
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "user-123" not in prompt
        assert "user_" in prompt  # hashed ID present

    def test_user_name_not_redacted(self):
        ctx = _make_context(user_id="user-123", user_name="Alice")
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "Alice" in prompt
        # user_id should not appear when user_name is present (name takes priority)
        assert "user-123" not in prompt

    def test_home_channel_id_hashed(self):
        hc = {
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="telegram:99999",
                name="Home Chat",
            )
        }
        ctx = _make_context(home_channels=hc)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "99999" not in prompt
        assert "telegram:" in prompt  # prefix preserved
        assert "Home Chat" in prompt  # name not redacted

    def test_home_channel_id_preserved_without_redaction(self):
        hc = {
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="telegram:99999",
                name="Home Chat",
            )
        }
        ctx = _make_context(home_channels=hc)
        prompt = build_session_context_prompt(ctx, redact_pii=False)
        assert "99999" in prompt

    def test_whatsapp_cloud_home_phone_fallback_name_is_redacted(self):
        wa_id = "15551234567"
        hc = {
            Platform.WHATSAPP_CLOUD: HomeChannel(
                platform=Platform.WHATSAPP_CLOUD,
                chat_id=wa_id,
                name=wa_id,
            )
        }
        ctx = _make_context(
            user_id=wa_id,
            chat_id=wa_id,
            chat_name=wa_id,
            platform=Platform.WHATSAPP_CLOUD,
            home_channels=hc,
        )

        prompt = build_session_context_prompt(ctx, redact_pii=True)

        assert wa_id not in prompt
        assert _hash_chat_id(wa_id) in prompt

    def test_ineligible_source_still_redacts_eligible_home_channel(self):
        phone = "15551234567"
        ctx = _make_context(
            user_id="discord-user-123",
            chat_id="discord-channel-456",
            platform=Platform.DISCORD,
            home_channels={
                Platform.WHATSAPP_CLOUD: HomeChannel(
                    platform=Platform.WHATSAPP_CLOUD,
                    chat_id=phone,
                    name=phone,
                )
            },
        )

        prompt = build_session_context_prompt(ctx, redact_pii=True)

        assert "discord-user-123" in prompt
        assert phone not in prompt
        assert _hash_chat_id(phone) in prompt

    def test_eligible_source_keeps_ineligible_home_channel_raw(self):
        discord_id = "discord-channel-456"
        ctx = _make_context(
            user_id="telegram-user-123",
            chat_id="telegram-chat-789",
            platform=Platform.TELEGRAM,
            home_channels={
                Platform.DISCORD: HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id=discord_id,
                    name=discord_id,
                )
            },
        )

        prompt = build_session_context_prompt(ctx, redact_pii=True)

        assert "telegram-user-123" not in prompt
        assert discord_id in prompt

    def test_redaction_is_deterministic(self):
        ctx = _make_context(user_id="+15551234567")
        prompt1 = build_session_context_prompt(ctx, redact_pii=True)
        prompt2 = build_session_context_prompt(ctx, redact_pii=True)
        assert prompt1 == prompt2

    def test_different_ids_produce_different_hashes(self):
        ctx1 = _make_context(user_id="user-A")
        ctx2 = _make_context(user_id="user-B")
        p1 = build_session_context_prompt(ctx1, redact_pii=True)
        p2 = build_session_context_prompt(ctx2, redact_pii=True)
        assert p1 != p2

    def test_discord_ids_not_redacted_even_with_flag(self):
        """Discord needs real IDs for <@user_id> mentions."""
        ctx = _make_context(user_id="123456789", platform=Platform.DISCORD)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "123456789" in prompt

    def test_whatsapp_ids_redacted(self):
        ctx = _make_context(user_id="+15551234567", platform=Platform.WHATSAPP)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "+15551234567" not in prompt
        assert "user_" in prompt

    def test_whatsapp_bridge_phone_fallback_name_is_redacted(self):
        phone = "15551234567"
        ctx = _make_context(
            user_id=f"{phone}@s.whatsapp.net",
            user_name=phone,
            chat_id=f"{phone}@s.whatsapp.net",
            chat_name=phone,
            platform=Platform.WHATSAPP,
        )
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert phone not in prompt
        assert "user_" in prompt

    def test_whatsapp_cloud_no_profile_name_redacts_wa_id_fallbacks(self):
        wa_id = "15551234567"
        ctx = _make_context(
            user_id=wa_id,
            user_name=None,
            chat_id=wa_id,
            chat_name=wa_id,
            platform=Platform.WHATSAPP_CLOUD,
        )
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert wa_id not in prompt
        assert "user_" in prompt

    def test_signal_ids_redacted(self):
        phone = "+15551234567"
        ctx = _make_context(
            user_id=phone,
            user_name=phone,
            chat_id=phone,
            chat_name=phone,
            platform=Platform.SIGNAL,
        )
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert phone not in prompt
        assert "user_" in prompt

    def test_bluebubbles_sender_address_fallback_is_redacted(self):
        phone = "+15551234567"
        chat_identifier = f"iMessage;-;{phone}"
        ctx = _make_context(
            user_id=phone,
            user_name=phone,
            chat_id=chat_identifier,
            chat_name=chat_identifier,
            platform=Platform.BLUEBUBBLES,
        )
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert phone not in prompt
        assert chat_identifier not in prompt
        assert "user_" in prompt

    def test_slack_ids_not_redacted(self):
        """Slack may need IDs for mentions too."""
        ctx = _make_context(user_id="U12345ABC", platform=Platform.SLACK)
        prompt = build_session_context_prompt(ctx, redact_pii=True)
        assert "U12345ABC" in prompt
