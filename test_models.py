#!/usr/bin/env python3
"""Test script for all AWS Bedrock Chat models."""

import time
import logging
from src.bedrock_chat.cli import chat_command
from src.bedrock_chat.models import get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(model_name: str, model_id: str):
    """Test a single model."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name} ({model_id})")
    print(f"{'='*80}")
    
    try:
        # Test basic text generation
        prompt = "Write a short poem about coding"
        print(f"\n1. Basic text generation")
        print(f"Prompt: '{prompt}'")
        response = chat_command(prompt, model_name, max_tokens=100)
        print(f"Response received ✓")
        
        # Add delay to avoid rate limits
        time.sleep(5)
        
        # Test creative writing
        prompt = "Tell me a very short story about a robot learning to paint"
        print(f"\n2. Creative writing")
        print(f"Prompt: '{prompt}'")
        response = chat_command(prompt, model_name, max_tokens=150, temperature=0.8)
        print(f"Response received ✓")
        
        # Add delay to avoid rate limits
        time.sleep(5)
        
        # Test analytical thinking
        prompt = "Explain the concept of recursion in programming in simple terms"
        print(f"\n3. Analytical thinking")
        print(f"Prompt: '{prompt}'")
        response = chat_command(prompt, model_name, max_tokens=200, temperature=0.3)
        print(f"Response received ✓")
        
        print(f"\nAll tests passed for {model_name} ✓")
        return True
        
    except Exception as e:
        logger.error(f"Error testing {model_name}: {str(e)}")
        return False

def main():
    """Main entry point."""
    print("\nStarting model tests...")
    models = get_available_models()
    
    results = {
        'claude': [],
        'titan': [],
        'llama': [],
        'mistral': []
    }
    
    # Group models by type
    for name, model_id in models.items():
        if name.startswith('claude-'):
            results['claude'].append((name, model_id))
        elif name.startswith('titan-'):
            results['titan'].append((name, model_id))
        elif name.startswith('llama-'):
            results['llama'].append((name, model_id))
        elif name.startswith('mistral-'):
            results['mistral'].append((name, model_id))
    
    # Test each model type
    success_count = 0
    total_count = len(models)
    
    for model_type, model_list in results.items():
        print(f"\n\n{'='*40}")
        print(f"Testing {model_type.title()} Models")
        print(f"{'='*40}")
        
        for name, model_id in model_list:
            if test_model(name, model_id):
                success_count += 1
            # Add longer delay between models
            time.sleep(10)
    
    # Print summary
    print(f"\n\n{'='*40}")
    print("Test Summary")
    print(f"{'='*40}")
    print(f"Total models tested: {total_count}")
    print(f"Successful tests: {success_count}")
    print(f"Failed tests: {total_count - success_count}")
    
    if success_count == total_count:
        print("\n✨ All models tested successfully!")
    else:
        print("\n⚠️ Some tests failed. Check the logs for details.")

if __name__ == '__main__':
    main() 