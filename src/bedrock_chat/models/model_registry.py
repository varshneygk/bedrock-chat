"""Model registry for AWS Bedrock Chat."""

from typing import Dict, Optional, Tuple, NamedTuple

class ModelInfo(NamedTuple):
    """Model information container."""
    model_id: str
    api_version: Optional[str]
    context_window: int
    supports_streaming: bool

# Map short names to model information
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # Claude Models
    'claude-sonnet': ModelInfo(
        'anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock-2023-05-31',
        200000,
        True
    ),
    'claude-haiku': ModelInfo(
        'anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock-2023-05-31',
        200000,
        True
    ),
    
    # Titan Models
    'titan-express': ModelInfo(
        'amazon.titan-text-express-v1',
        None,
        8000,
        True
    ),
    'titan-lite': ModelInfo(
        'amazon.titan-text-lite-v1',
        None,
        4000,
        True
    ),
    
    # Llama Models
    'llama-70b': ModelInfo(
        'meta.llama3-70b-instruct-v1:0',
        None,
        32000,
        True
    ),
    'llama-8b': ModelInfo(
        'meta.llama3-8b-instruct-v1:0',
        None,
        16000,
        True
    ),
    
    # Mistral Models
    'mistral-7b': ModelInfo(
        'mistral.mistral-7b-instruct-v0:2',
        None,
        8000,
        True
    ),
    'mistral-large': ModelInfo(
        'mistral.mistral-large-2402-v1:0',
        None,
        32000,
        True
    ),
    'mistral-8x7b': ModelInfo(
        'mistral.mixtral-8x7b-instruct-v0:1',
        None,
        32000,
        True
    )
}

def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get model information from short name.
    
    Args:
        model_name: Short name of the model (e.g. 'claude-sonnet', 'titan-express')
        
    Returns:
        ModelInfo if found, None otherwise
    """
    return MODEL_REGISTRY.get(model_name)

def get_model_id(model_name: str) -> Optional[str]:
    """Get full model ID from short name.
    
    Args:
        model_name: Short name of the model (e.g. 'claude-sonnet', 'titan-express')
        
    Returns:
        Full model ID if found, None otherwise
    """
    model_info = get_model_info(model_name)
    return model_info.model_id if model_info else None

def get_available_models() -> Dict[str, str]:
    """Get all available models with their IDs.
    
    Returns:
        Dictionary mapping short names to full model IDs
    """
    return {name: info.model_id for name, info in MODEL_REGISTRY.items()}
