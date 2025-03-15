#!/usr/bin/env python3
"""Streaming chat script for AWS Bedrock Chat."""

import sys
import time
from src.bedrock_chat.cli import stream_chat_command

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python chat_stream.py \"<prompt>\" <model> [max_tokens] [temperature]")
        print("\nExample: python chat_stream.py \"Tell me a joke\" claude-haiku 100 0.7")
        sys.exit(1)
        
    prompt = sys.argv[1]
    model = sys.argv[2]
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    
    start_time = time.time()
    stream_chat_command(prompt, model, max_tokens, temperature)
    print(f"\nChat completed in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
