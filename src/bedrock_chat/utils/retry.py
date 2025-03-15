"""Retry and rate limiting utilities."""

import random
import time
from typing import Optional, Callable
from ..models.model_config import StreamConfig

def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with jitter.
    
    Args:
        attempt: Current retry attempt number
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay *= (0.5 + random.random())
    return delay

def handle_rate_limit(
    attempt: int,
    config: StreamConfig,
    print_fn: Optional[Callable[[str], None]] = print
) -> None:
    """Handle rate limiting with exponential backoff.
    
    Args:
        attempt: Current retry attempt
        config: Stream configuration
        print_fn: Optional function to print status messages
    """
    delay = calculate_backoff_delay(
        attempt,
        base_delay=config.base_delay,
        max_delay=config.max_delay
    )
    
    if print_fn:
        print_fn(f"\n⚠️ Rate limited by AWS. Waiting {delay:.1f}s before retry {attempt}/{config.retry_attempts}...")
    
    time.sleep(delay)
