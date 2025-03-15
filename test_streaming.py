#!/usr/bin/env python3
"""Test script for streaming functionality across AWS Bedrock Chat models."""

import time
import logging
from src.bedrock_chat.cli import stream_chat_command
from src.bedrock_chat.models import get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_streaming(model_name: str, model_id: str):
    """Test streaming for a single model."""
    print(f"\n{'='*80}")
    print(f"Testing streaming for {model_name} ({model_id})")
    print(f"{'='*80}")
    
    try:
        # Test basic streaming
        prompt = "Write a story about an AI learning to dance, one sentence at a time"
        print(f"\n1. Basic streaming test")
        print(f"Prompt: '{prompt}'")
        response = stream_chat_command(prompt, model_name, max_tokens=150)
        print(f"\nStreaming test completed ✓")
        
        # Add delay to avoid rate limits
        time.sleep(5)
        
        # Test streaming with system prompt
        system_prompt = "You are a creative storyteller who writes engaging narratives"
        prompt = "Tell me about a magical forest, revealing details gradually"
        print(f"\n2. Streaming with system prompt")
        print(f"System: '{system_prompt}'")
        print(f"Prompt: '{prompt}'")
        response = stream_chat_command(
            prompt, 
            model_name, 
            max_tokens=200,
            system_prompt=system_prompt,
            temperature=0.8
        )
        print(f"\nStreaming with system prompt completed ✓")
        
        # Add delay to avoid rate limits
        time.sleep(5)
        
        # Test streaming with chat history
        from src.bedrock_chat.cli import ChatHistory
        chat_history = ChatHistory()
        chat_history.add_message("user", "Let's write a story together about a time traveler")
        chat_history.add_message("assistant", "I'd love to help write a story about a time traveler. Where should we begin?")
        
        prompt = "The time traveler discovers something unexpected in the year 2150"
        print(f"\n3. Streaming with chat history")
        print(f"Prompt: '{prompt}'")
        response = stream_chat_command(
            prompt,
            model_name,
            max_tokens=200,
            chat_history=chat_history,
            temperature=0.7
        )
        print(f"\nStreaming with chat history completed ✓")
        
        print(f"\nAll streaming tests passed for {model_name} ✓")
        return True
        
    except Exception as e:
        logger.error(f"Error testing streaming for {model_name}: {str(e)}")
        return False

def main():
    """Main entry point."""
    print("\nStarting streaming tests...")
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
        print(f"Testing {model_type.title()} Models Streaming")
        print(f"{'='*40}")
        
        for name, model_id in model_list:
            if test_model_streaming(name, model_id):
                success_count += 1
            # Add longer delay between models
            time.sleep(15)  # Increased delay for streaming tests
    
    # Print summary
    print(f"\n\n{'='*40}")
    print("Streaming Test Summary")
    print(f"{'='*40}")
    print(f"Total models tested: {total_count}")
    print(f"Successful tests: {success_count}")
    print(f"Failed tests: {total_count - success_count}")
    
    if success_count == total_count:
        print("\n✨ All streaming tests completed successfully!")
    else:
        print("\n⚠️ Some streaming tests failed. Check the logs for details.")

if __name__ == '__main__':
    main() 