#!/usr/bin/env python3
"""List available AWS Bedrock Chat models."""

from src.bedrock_chat.cli import list_models_command

if __name__ == '__main__':
    list_models_command()
