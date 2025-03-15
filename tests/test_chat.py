"""Tests for the chat functionality."""

import pytest
import json
from unittest.mock import Mock, patch
from src.bedrock_chat.cli.chat import (
    ChatHistory,
    format_prompt,
    chat_command,
    stream_chat_command
)

@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client."""
    with patch('src.bedrock_chat.cli.chat.get_bedrock_client') as mock_client:
        # Mock non-streaming response
        mock_response = Mock()
        mock_response.get.return_value.read.return_value = '{"completion": "Test response"}'
        mock_client.return_value.invoke_model.return_value = mock_response
        
        # Mock streaming response
        mock_stream = Mock()
        mock_stream.get.return_value = [
            {'chunk': {'bytes': json.dumps({"completion": "Test"}).encode()}},
            {'chunk': {'bytes': json.dumps({"completion": " response"}).encode()}}
        ]
        mock_client.return_value.invoke_model_with_response_stream.return_value = mock_stream
        
        yield mock_client

def test_chat_history():
    """Test chat history functionality."""
    history = ChatHistory()
    
    # Test adding messages
    history.add_message("user", "Hello")
    history.add_message("assistant", "Hi there!")
    assert len(history.messages) == 2
    assert history.messages[0]["role"] == "user"
    assert history.messages[0]["content"] == "Hello"
    
    # Test getting context
    context = history.get_context(max_messages=1)
    assert len(context) == 1
    assert context[0]["role"] == "assistant"
    
    # Test clearing history
    history.clear()
    assert len(history.messages) == 0

def test_format_prompt():
    """Test prompt formatting."""
    # Test basic prompt
    assert format_prompt("Hello") == "Hello"
    
    # Test with system prompt
    formatted = format_prompt("Hello", system_prompt="Be helpful")
    assert isinstance(formatted, list)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"
    
    # Test with chat history
    history = ChatHistory()
    history.add_message("user", "Previous message")
    history.add_message("assistant", "Previous response")
    
    formatted = format_prompt(
        "Hello",
        system_prompt="Be helpful",
        chat_history=history
    )
    assert isinstance(formatted, list)
    assert len(formatted) == 4  # system + 2 history + current
    assert formatted[0]["role"] == "system"
    assert formatted[-1]["content"] == "Hello"

@pytest.mark.asyncio
async def test_chat_command(mock_bedrock_client):
    """Test non-streaming chat command."""
    # Test basic chat
    response = chat_command(
        prompt="Hello",
        model_name="claude-sonnet",
        max_tokens=100,
        temperature=0.7
    )
    assert response == "Test response"
    
    # Test with system prompt and history
    history = ChatHistory()
    response = chat_command(
        prompt="Hello",
        model_name="claude-sonnet",
        system_prompt="Be helpful",
        chat_history=history
    )
    assert response == "Test response"
    assert len(history.messages) == 1
    
    # Test error handling
    mock_bedrock_client.return_value.invoke_model.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        chat_command("Hello", "claude-sonnet")

@pytest.mark.asyncio
async def test_stream_chat_command(mock_bedrock_client):
    """Test streaming chat command."""
    # Test basic streaming
    response = stream_chat_command(
        prompt="Hello",
        model_name="claude-sonnet",
        max_tokens=100,
        temperature=0.7,
        print_fn=lambda x: None  # Suppress output during test
    )
    assert response == "Test response"
    
    # Test with system prompt and history
    history = ChatHistory()
    response = stream_chat_command(
        prompt="Hello",
        model_name="claude-sonnet",
        system_prompt="Be helpful",
        chat_history=history,
        print_fn=lambda x: None  # Suppress output during test
    )
    assert response == "Test response"
    assert len(history.messages) == 1
    
    # Test error handling
    mock_bedrock_client.return_value.invoke_model_with_response_stream.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        stream_chat_command("Hello", "claude-sonnet")

def test_invalid_model():
    """Test handling of invalid models."""
    with pytest.raises(ValueError, match="Unknown model"):
        chat_command("Hello", "invalid-model")
        
    with pytest.raises(ValueError, match="Unknown model"):
        stream_chat_command("Hello", "invalid-model") 