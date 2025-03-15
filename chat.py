#!/usr/bin/env python3
"""Non-streaming chat script for AWS Bedrock Chat."""

import sys
import time
import logging
from typing import Optional
from src.bedrock_chat.cli import chat_command
from src.bedrock_chat.models import get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_available_models():
    """List all available models."""
    models = get_available_models()
    print("\nAvailable models:")
    for name, model_id in models.items():
        print(f"- {name:<15} ({model_id})")
    print()

def validate_args(args: list) -> tuple[str, str, Optional[int], Optional[float]]:
    """Validate command line arguments."""
    if len(args) < 2 or args[1] in ['-h', '--help']:
        print("Usage: python chat.py \"<prompt>\" <model> [max_tokens] [temperature]")
        print("\nExample: python chat.py \"Tell me a joke\" claude-haiku 100 0.7")
        print("\nUse --list to see available models")
        sys.exit(0)
        
    if args[1] == '--list':
        list_available_models()
        sys.exit(0)
        
    if len(args) < 3:
        print("Error: Both prompt and model name are required")
        print("Use --help for usage information")
        sys.exit(1)
        
    prompt = args[1]
    model = args[2]
    
    try:
        max_tokens = int(args[3]) if len(args) > 3 else 100
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
    except ValueError as e:
        print(f"Error: Invalid max_tokens value - {str(e)}")
        sys.exit(1)
        
    try:
        temperature = float(args[4]) if len(args) > 4 else 0.7
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
    except ValueError as e:
        print(f"Error: Invalid temperature value - {str(e)}")
        sys.exit(1)
        
    return prompt, model, max_tokens, temperature

def main():
    """Main entry point."""
    try:
        # Parse and validate arguments
        prompt, model, max_tokens, temperature = validate_args(sys.argv)
        
        logger.info("Starting chat request")
        logger.info(f"Model: {model}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Temperature: {temperature}")
        
        # Run chat command
        start_time = time.time()
        response = chat_command(prompt, model, max_tokens, temperature)
        elapsed = time.time() - start_time
        
        # Print response with formatting
        print("\n" + "="*80)
        print("Response:")
        print("-"*80)
        print(response)
        print("="*80)
        print(f"\nChat completed in {elapsed:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
