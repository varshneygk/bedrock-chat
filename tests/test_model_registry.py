"""Tests for the model registry module."""

import pytest
from src.bedrock_chat.models.model_registry import (
    get_model_info,
    get_model_id,
    get_available_models,
    ModelInfo
)

def test_get_model_info_valid():
    """Test getting model info for valid models."""
    # Test Claude model
    info = get_model_info('claude-sonnet')
    assert info is not None
    assert info.model_id == 'anthropic.claude-3-sonnet-20240229-v1:0'
    assert info.api_version == 'bedrock-2023-05-31'
    assert info.context_window == 200000
    assert info.supports_streaming is True

    # Test Titan model
    info = get_model_info('titan-express')
    assert info is not None
    assert info.model_id == 'amazon.titan-text-express-v1'
    assert info.api_version is None
    assert info.context_window == 8000
    assert info.supports_streaming is True

    # Test Llama model
    info = get_model_info('llama-70b')
    assert info is not None
    assert info.model_id == 'meta.llama3-70b-instruct-v1:0'
    assert info.api_version is None
    assert info.context_window == 32000
    assert info.supports_streaming is True

    # Test Mistral model
    info = get_model_info('mistral-large')
    assert info is not None
    assert info.model_id == 'mistral.mistral-large-2402-v1:0'
    assert info.api_version is None
    assert info.context_window == 32000
    assert info.supports_streaming is True

def test_get_model_info_invalid():
    """Test getting model info for invalid models."""
    assert get_model_info('nonexistent-model') is None
    assert get_model_info('') is None
    assert get_model_info('invalid-name') is None

def test_get_model_id_valid():
    """Test getting model ID for valid models."""
    assert get_model_id('claude-sonnet') == 'anthropic.claude-3-sonnet-20240229-v1:0'
    assert get_model_id('titan-express') == 'amazon.titan-text-express-v1'
    assert get_model_id('llama-70b') == 'meta.llama3-70b-instruct-v1:0'
    assert get_model_id('mistral-large') == 'mistral.mistral-large-2402-v1:0'

def test_get_model_id_invalid():
    """Test getting model ID for invalid models."""
    assert get_model_id('nonexistent-model') is None
    assert get_model_id('') is None
    assert get_model_id('invalid-name') is None

def test_get_available_models():
    """Test getting all available models."""
    models = get_available_models()
    assert isinstance(models, dict)
    assert len(models) > 0
    
    # Check for presence of models from each provider
    assert any(id_.startswith('anthropic.') for id_ in models.values())
    assert any(id_.startswith('amazon.') for id_ in models.values())
    assert any(id_.startswith('meta.') for id_ in models.values())
    assert any(id_.startswith('mistral.') for id_ in models.values())

    # Check specific models
    assert models['claude-sonnet'] == 'anthropic.claude-3-sonnet-20240229-v1:0'
    assert models['titan-express'] == 'amazon.titan-text-express-v1'
    assert models['llama-70b'] == 'meta.llama3-70b-instruct-v1:0'
    assert models['mistral-large'] == 'mistral.mistral-large-2402-v1:0' 