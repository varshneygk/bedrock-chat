"""Examples of using different models with AWS Bedrock Chat."""

from src.bedrock_chat.cli import chat_command, stream_chat_command
from src.bedrock_chat.models import ModelConfig, StreamConfig

def claude_examples():
    """Examples using Claude models."""
    print("\n=== Claude Models Examples ===")
    
    # Basic chat with Claude Sonnet
    print("\n1. Basic Chat with Claude Sonnet:")
    chat_command(
        prompt="Explain what is quantum computing in simple terms",
        model_name="claude-sonnet"
    )
    
    # Streaming chat with Claude Haiku
    print("\n2. Streaming Chat with Claude Haiku:")
    stream_chat_command(
        prompt="Write a haiku about spring",
        model_name="claude-haiku",
        max_tokens=50,
        temperature=0.8
    )
    
    # Long-form content with Claude 2.1
    print("\n3. Long-form Content with Claude 2.1:")
    chat_command(
        prompt="Write a detailed analysis of climate change impacts",
        model_name="claude-2.1",
        max_tokens=1000,
        temperature=0.7
    )

def titan_examples():
    """Examples using Amazon Titan models."""
    print("\n=== Titan Models Examples ===")
    
    # Quick response with Titan Express
    print("\n1. Quick Response with Titan Express:")
    chat_command(
        prompt="Give me 3 quick tips for productivity",
        model_name="titan-express",
        max_tokens=100
    )
    
    # Creative writing with Titan Lite
    print("\n2. Creative Writing with Titan Lite:")
    stream_chat_command(
        prompt="Write a short story about a robot learning to paint",
        model_name="titan-lite",
        max_tokens=200,
        temperature=0.9
    )

def llama_examples():
    """Examples using Llama models."""
    print("\n=== Llama Models Examples ===")
    
    # Complex reasoning with Llama 70B
    print("\n1. Complex Reasoning with Llama 70B:")
    chat_command(
        prompt="Explain the concept of blockchain and its potential applications",
        model_name="llama-70b",
        max_tokens=500
    )
    
    # Creative task with Llama 8B
    print("\n2. Creative Task with Llama 8B:")
    stream_chat_command(
        prompt="Create a fantasy character description",
        model_name="llama-8b",
        max_tokens=150,
        temperature=0.85
    )

def mistral_examples():
    """Examples using Mistral models."""
    print("\n=== Mistral Models Examples ===")
    
    # Technical explanation with Mistral Large
    print("\n1. Technical Explanation with Mistral Large:")
    chat_command(
        prompt="Explain how neural networks learn and adapt",
        model_name="mistral-large",
        max_tokens=300
    )
    
    # Code generation with Mistral 7B
    print("\n2. Code Generation with Mistral 7B:")
    stream_chat_command(
        prompt="Write a Python function to calculate Fibonacci numbers",
        model_name="mistral-7b",
        max_tokens=200,
        temperature=0.3
    )
    
    # Multi-task with Mistral 8x7B
    print("\n3. Multi-task with Mistral 8x7B:")
    chat_command(
        prompt="1. Summarize the theory of relativity\n2. List its key equations\n3. Explain practical applications",
        model_name="mistral-8x7b",
        max_tokens=800
    )

def advanced_examples():
    """Advanced usage examples."""
    print("\n=== Advanced Usage Examples ===")
    
    # Custom configuration
    print("\n1. Custom Configuration:")
    config = ModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.9,
        max_tokens=2000,
        top_p=0.95,
        top_k=100,
        stop_sequences=["\n###\n"]
    )
    
    # Custom streaming configuration
    stream_config = StreamConfig(
        chunk_size=2048,
        retry_attempts=3,
        base_delay=1.5,
        max_delay=20.0,
        throttle_cooldown=5.0
    )
    
    # Error handling example
    print("\n2. Error Handling Example:")
    try:
        chat_command(
            prompt="Generate a very long response",
            model_name="claude-sonnet",
            max_tokens=1000000  # Intentionally too large
        )
    except Exception as e:
        print(f"Caught error as expected: {e}")

if __name__ == "__main__":
    print("Running AWS Bedrock Chat Examples...")
    
    # Run individual model examples
    claude_examples()
    titan_examples()
    llama_examples()
    mistral_examples()
    
    # Run advanced examples
    advanced_examples()
    
    print("\nAll examples completed!") 