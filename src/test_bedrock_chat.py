"""Test suite for AWS Bedrock Chat."""

import pytest
from bedrock_chat.models.model_registry import get_model_id
from bedrock_chat.utils.client import get_bedrock_client
from bedrock_chat.cli.chat import chat_command, stream_chat_command


class TestModelRegistry:
    def test_get_model_id(self):
        assert get_model_id('claude-haiku') == 'anthropic.claude-3-haiku-20240307-v1:0'
        assert get_model_id('titan-express') == 'amazon.titan-text-express-v1'
        assert get_model_id('unknown-model') is None


class TestClient:
    def test_get_bedrock_client(self):
        client = get_bedrock_client()
        assert client is not None


class TestChatCommands:
    def test_chat_command(self):
        response = chat_command('Tell me a joke', 'claude-haiku', 100, 0.7)
        assert response is not None

    def test_stream_chat_command(self):
        response = stream_chat_command('Tell me a joke', 'claude-haiku', 100, 0.7)
        assert response is not None
