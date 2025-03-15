# AWS Bedrock Chat

A Python library for interacting with AWS Bedrock models, featuring both streaming and non-streaming implementations.

## Features

- Simple short names for models (e.g. 'claude-haiku' instead of full ARNs)
- Streaming and non-streaming chat implementations
- Automatic rate limiting and retry handling
- Support for Claude, Titan, Llama, and Mistral models

## Project Structure

```
bedrock-chat/
├── src/
│   └── bedrock_chat/
│       ├── models/          # Model configurations and registry
│       ├── utils/           # AWS client, retry, and streaming utilities
│       └── cli/            # Command-line interface implementations
├── chat.py                 # Non-streaming chat script
├── chat_stream.py         # Streaming chat script
└── list_models.py         # Model listing script
```

## Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials in `.env`:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

## Usage

1. List available models:
```bash
python list_models.py
```

2. Chat with a model (non-streaming):
```bash
python chat.py "Your prompt" <model> [max_tokens] [temperature]

# Example
python chat.py "Tell me a joke" claude-haiku 100 0.7
```

3. Chat with a model (streaming):
```bash
python chat_stream.py "Your prompt" <model> [max_tokens] [temperature]

# Example
python chat_stream.py "Tell me a joke" claude-haiku 100 0.7
```

## Available Models

1. Claude Models:
- `claude-sonnet` - Claude 3 Sonnet (most capable)
- `claude-haiku` - Claude 3 Haiku (faster)

2. Titan Models:
- `titan-express` - Titan Text Express
- `titan-lite` - Titan Text Lite

3. Llama Models:
- `llama-70b` - Llama 3 70B
- `llama-8b` - Llama 3 8B

4. Mistral Models:
- `mistral-7b` - Mistral 7B
- `mistral-large` - Mistral Large
- `mistral-8x7b` - Mixtral 8x7B

## Model Selection Guide

- Complex reasoning: Use `claude-sonnet`
- Quick responses: Use `claude-haiku`
- Simple Q&A: Use `titan-express`
- Creative tasks: Use `llama-70b`
