"""Model listing command for AWS Bedrock Chat CLI."""

from ..models import get_available_models

def list_models_command() -> None:
    """List all available models with their IDs."""
    models = get_available_models()
    
    print("\nAvailable Models:")
    print("================\n")
    
    # Claude Models
    print("Claude Models:")
    print("-------------")
    for name, id in models.items():
        if name.startswith('claude-'):
            print(f"- {name}: {id}")
    
    # Titan Models
    print("\nTitan Models:")
    print("------------")
    for name, id in models.items():
        if name.startswith('titan-'):
            print(f"- {name}: {id}")
    
    # Llama Models
    print("\nLlama Models:")
    print("------------")
    for name, id in models.items():
        if name.startswith('llama-'):
            print(f"- {name}: {id}")
    
    # Mistral Models
    print("\nMistral Models:")
    print("--------------")
    for name, id in models.items():
        if name.startswith('mistral-'):
            print(f"- {name}: {id}")
    
    print("\nUsage Examples:")
    print("--------------")
    print("# Streaming chat")
    print("python chat_stream.py \"Your prompt\" claude-haiku 100 0.7")
    print("\n# Non-streaming chat")
    print("python chat.py \"Your prompt\" titan-express 100 0.7\n")
