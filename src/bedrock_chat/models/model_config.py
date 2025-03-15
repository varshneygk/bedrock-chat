"""Model configuration classes for AWS Bedrock Chat."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .model_registry import get_model_info, ModelInfo

@dataclass
class ModelConfig:
    """Configuration for a Bedrock model."""
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 100
    top_p: float = 1.0
    top_k: int = 250
    stop_sequences: Optional[list[str]] = None
    api_version: Optional[str] = None
    model_info: Optional[ModelInfo] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top-p must be between 0 and 1")
        if self.top_k < 0:
            raise ValueError("Top-k must be non-negative")
        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")
        
    @classmethod
    def from_model_name(cls, model_name: str, **kwargs) -> 'ModelConfig':
        """Create config from model name.
        
        Args:
            model_name: Short model name (e.g. 'claude-haiku')
            **kwargs: Additional config parameters
            
        Returns:
            ModelConfig instance
            
        Raises:
            ValueError: If model name is unknown or parameters are invalid
        """
        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Ensure max_tokens doesn't exceed model's context window
        max_tokens = kwargs.get('max_tokens', 100)
        if max_tokens > model_info.context_window:
            raise ValueError(
                f"Max tokens ({max_tokens}) exceeds model's context window "
                f"({model_info.context_window})"
            )
            
        return cls(
            model_id=model_info.model_id,
            api_version=model_info.api_version,
            model_info=model_info,
            **kwargs
        )
    
    def to_request_body(self) -> Dict[str, Any]:
        """Convert config to request body based on model type.
        
        Returns:
            Dictionary containing model-specific request parameters
            
        Raises:
            ValueError: If model type is unsupported
        """
        if self.model_id.startswith('anthropic.claude-3'):
            return {
                "anthropic_version": self.api_version or "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "stop_sequences": self.stop_sequences or []
            }
        elif self.model_id.startswith('anthropic.'):
            return {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "stop_sequences": self.stop_sequences or []
            }
        elif self.model_id.startswith('amazon.'):
            return {
                "inputText": "",  # Will be set in chat command
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": self.top_p,
                    "stopSequences": self.stop_sequences or []
                }
            }
        elif self.model_id.startswith('meta.'):
            return {
                "max_gen_len": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        elif self.model_id.startswith('mistral.'):
            return {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": self.stop_sequences or []
            }
        else:
            raise ValueError(f"Unsupported model: {self.model_id}")

@dataclass
class StreamConfig:
    """Configuration for streaming responses."""
    chunk_size: int = 1024
    retry_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 30.0
    throttle_cooldown: float = 10.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.chunk_size < 1:
            raise ValueError("Chunk size must be positive")
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("Base delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("Max delay must be greater than base delay")
        if self.throttle_cooldown < 0:
            raise ValueError("Throttle cooldown must be non-negative")
