"""Tests for the model configuration module."""

import pytest
from src.bedrock_chat.models.model_config import ModelConfig, StreamConfig

def test_model_config_validation():
    """Test model configuration parameter validation."""
    # Test valid configuration
    config = ModelConfig(model_id='test.model')
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 1.0
    assert config.top_k == 250
    assert config.stop_sequences is None

    # Test invalid temperature
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        ModelConfig(model_id='test.model', temperature=1.5)
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        ModelConfig(model_id='test.model', temperature=-0.1)

    # Test invalid top_p
    with pytest.raises(ValueError, match="Top-p must be between 0 and 1"):
        ModelConfig(model_id='test.model', top_p=1.5)
    with pytest.raises(ValueError, match="Top-p must be between 0 and 1"):
        ModelConfig(model_id='test.model', top_p=-0.1)

    # Test invalid top_k
    with pytest.raises(ValueError, match="Top-k must be non-negative"):
        ModelConfig(model_id='test.model', top_k=-1)

    # Test invalid max_tokens
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        ModelConfig(model_id='test.model', max_tokens=0)
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        ModelConfig(model_id='test.model', max_tokens=-1)

def test_model_config_from_model_name():
    """Test creating model config from model name."""
    # Test valid model
    config = ModelConfig.from_model_name('claude-sonnet')
    assert config.model_id == 'anthropic.claude-3-sonnet-20240229-v1:0'
    assert config.api_version == 'bedrock-2023-05-31'
    assert config.model_info is not None
    assert config.model_info.context_window == 200000

    # Test invalid model
    with pytest.raises(ValueError, match="Unknown model: invalid-model"):
        ModelConfig.from_model_name('invalid-model')

    # Test exceeding context window
    with pytest.raises(ValueError, match="Max tokens .* exceeds model's context window"):
        ModelConfig.from_model_name('claude-sonnet', max_tokens=250000)

def test_model_config_request_body():
    """Test generating request body for different model types."""
    # Test Claude model
    config = ModelConfig.from_model_name('claude-sonnet')
    body = config.to_request_body()
    assert body['anthropic_version'] == 'bedrock-2023-05-31'
    assert body['max_tokens'] == 100
    assert body['temperature'] == 0.7
    assert body['top_k'] == 250
    assert body['top_p'] == 1.0
    assert body['stop_sequences'] == []

    # Test Titan model
    config = ModelConfig.from_model_name('titan-express')
    body = config.to_request_body()
    assert body['maxTokenCount'] == 100
    assert body['temperature'] == 0.7
    assert body['topP'] == 1.0
    assert body['stopSequences'] == []

    # Test Llama model
    config = ModelConfig.from_model_name('llama-70b')
    body = config.to_request_body()
    assert body['max_gen_len'] == 100
    assert body['temperature'] == 0.7
    assert body['top_p'] == 1.0

    # Test Mistral model
    config = ModelConfig.from_model_name('mistral-large')
    body = config.to_request_body()
    assert body['max_tokens'] == 100
    assert body['temperature'] == 0.7
    assert body['top_p'] == 1.0
    assert body['stop'] == []

    # Test unsupported model
    config = ModelConfig(model_id='unsupported.model')
    with pytest.raises(ValueError, match="Unsupported model: unsupported.model"):
        config.to_request_body()

def test_stream_config_validation():
    """Test stream configuration parameter validation."""
    # Test valid configuration
    config = StreamConfig()
    assert config.chunk_size == 1024
    assert config.retry_attempts == 5
    assert config.base_delay == 2.0
    assert config.max_delay == 30.0
    assert config.throttle_cooldown == 10.0

    # Test invalid chunk size
    with pytest.raises(ValueError, match="Chunk size must be positive"):
        StreamConfig(chunk_size=0)
    with pytest.raises(ValueError, match="Chunk size must be positive"):
        StreamConfig(chunk_size=-1)

    # Test invalid retry attempts
    with pytest.raises(ValueError, match="Retry attempts must be non-negative"):
        StreamConfig(retry_attempts=-1)

    # Test invalid base delay
    with pytest.raises(ValueError, match="Base delay must be positive"):
        StreamConfig(base_delay=0)
    with pytest.raises(ValueError, match="Base delay must be positive"):
        StreamConfig(base_delay=-1)

    # Test invalid max delay
    with pytest.raises(ValueError, match="Max delay must be greater than base delay"):
        StreamConfig(base_delay=2.0, max_delay=1.0)

    # Test invalid throttle cooldown
    with pytest.raises(ValueError, match="Throttle cooldown must be non-negative"):
        StreamConfig(throttle_cooldown=-1) 