"""Chat commands for AWS Bedrock Chat CLI."""

import json
import sys
import time
from typing import Optional, Dict, Any, List, Union
from botocore.exceptions import ClientError

from ..models import ModelConfig, StreamConfig, get_model_id, get_model_info
from ..utils import get_bedrock_client, process_stream_chunks, handle_rate_limit

class ChatHistory:
    """Maintains chat history for context."""
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({"role": role, "content": content})
        
    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent message context."""
        return self.messages[-max_messages:]
    
    def clear(self):
        """Clear chat history."""
        self.messages = []

def format_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    chat_history: Optional[ChatHistory] = None
) -> Union[str, List[Dict[str, str]]]:
    """Format prompt based on model requirements.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system instructions
        chat_history: Optional chat history
        
    Returns:
        Formatted prompt (string or message list)
    """
    if chat_history and chat_history.messages:
        messages = chat_history.get_context()
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    elif system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        return prompt

def chat_command(
    prompt: str,
    model_name: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    chat_history: Optional[ChatHistory] = None,
    stream: bool = False
) -> str:
    """Run chat command with enhanced features.
    
    Args:
        prompt: User prompt
        model_name: Short model name (e.g. 'claude-haiku')
        max_tokens: Maximum tokens to generate
        temperature: Temperature for response generation
        system_prompt: Optional system instructions
        chat_history: Optional chat history for context
        stream: Whether to use streaming mode
        
    Returns:
        Model response text
    """
    try:
        # Get model info and create config
        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model '{model_name}'")
            
        config = ModelConfig.from_model_name(
            model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Format prompt
        formatted_prompt = format_prompt(prompt, system_prompt, chat_history)
        
        print(f"\nModel: {model_name}")
        print(f"Prompt: '{prompt}'")
        if system_prompt:
            print(f"System: '{system_prompt}'")
        print("Starting request...\n")
        
        start_time = time.time()
        
        # Make API call
        client = get_bedrock_client(streaming=stream)
        
        if stream:
            return stream_chat_command(
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                chat_history=chat_history
            )
            
        # Prepare request body based on model type
        if model_info.model_id.startswith('anthropic.claude-3'):
            body = {
                "messages": formatted_prompt if isinstance(formatted_prompt, list) else [{"role": "user", "content": formatted_prompt}],
                **config.to_request_body()
            }
        elif model_info.model_id.startswith('anthropic.'):
            body = {
                "prompt": formatted_prompt if isinstance(formatted_prompt, str) else "\n".join(msg["content"] for msg in formatted_prompt),
                **config.to_request_body()
            }
        elif model_info.model_id.startswith('amazon.'):
            # For Titan models, combine prompt with context
            final_prompt = formatted_prompt
            if isinstance(formatted_prompt, list):
                final_prompt = "\n".join(msg["content"] for msg in formatted_prompt)
            
            body = config.to_request_body()
            body["inputText"] = final_prompt
        elif model_info.model_id.startswith('meta.'):
            # For Llama models, add system prompt and format
            if system_prompt:
                prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                prompt = f"[INST] {prompt} [/INST]"
                
            body = {
                "prompt": prompt,
                **config.to_request_body()
            }
        else:
            # For other models, concatenate context if available
            final_prompt = formatted_prompt
            if isinstance(formatted_prompt, list):
                final_prompt = "\n".join(msg["content"] for msg in formatted_prompt)
            
            body = {
                "prompt": final_prompt,
                **config.to_request_body()
            }
            
        response = client.invoke_model(
            modelId=model_info.model_id,
            body=json.dumps(body)
        )
        
        # Parse response
        response_body = json.loads(response.get('body').read())
        response_text = ""
        
        if model_info.model_id.startswith('anthropic.claude-3'):
            response_text = response_body.get('content', [{}])[0].get('text', '')
        elif model_info.model_id.startswith('anthropic.'):
            response_text = response_body.get('completion', '')
        elif model_info.model_id.startswith('amazon.'):
            response_text = response_body.get('results', [{}])[0].get('outputText', '')
        elif model_info.model_id.startswith('meta.'):
            response_text = response_body.get('generation', '')
        else:
            response_text = response_body.get('outputs', [{}])[0].get('text', '')
            
        # Update chat history if available
        if chat_history is not None:
            chat_history.add_message("assistant", response_text)
            
        print(f"Response: {response_text}")
        print(f"\nRequest completed in {time.time() - start_time:.2f} seconds")
        
        return response_text
            
    except ClientError as e:
        error_msg = f"\n❌ AWS API error: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"\n❌ Error: {str(e)}"
        print(error_msg)
        raise

def stream_chat_command(
    prompt: str,
    model_name: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    chat_history: Optional[ChatHistory] = None,
    print_fn: Optional[callable] = print
) -> str:
    """Run streaming chat command with enhanced features.
    
    Args:
        prompt: User prompt
        model_name: Short model name (e.g. 'claude-haiku')
        max_tokens: Maximum tokens to generate
        temperature: Temperature for response generation
        system_prompt: Optional system instructions
        chat_history: Optional chat history
        print_fn: Optional function to print status messages
        
    Returns:
        Complete response text
    """
    try:
        # Get model info and create configs
        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model '{model_name}'")
            
        model_config = ModelConfig.from_model_name(
            model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        stream_config = StreamConfig()
        
        # Format prompt
        formatted_prompt = format_prompt(prompt, system_prompt, chat_history)
        
        print(f"\nModel: {model_name}")
        print(f"Prompt: '{prompt}'")
        if system_prompt:
            print(f"System: '{system_prompt}'")
        print("Starting streaming request...\n")
        
        start_time = time.time()
        
        # Prepare request body based on model type
        if model_info.model_id.startswith('anthropic.claude-3'):
            body = {
                "messages": formatted_prompt if isinstance(formatted_prompt, list) else [{"role": "user", "content": formatted_prompt}],
                **model_config.to_request_body()
            }
        elif model_info.model_id.startswith('anthropic.'):
            body = {
                "prompt": formatted_prompt if isinstance(formatted_prompt, str) else "\n".join(msg["content"] for msg in formatted_prompt),
                **model_config.to_request_body()
            }
        elif model_info.model_id.startswith('amazon.'):
            # For Titan models, combine prompt with context
            final_prompt = formatted_prompt
            if isinstance(formatted_prompt, list):
                final_prompt = "\n".join(msg["content"] for msg in formatted_prompt)
            
            body = model_config.to_request_body()
            body["inputText"] = final_prompt
        elif model_info.model_id.startswith('meta.'):
            # For Llama models, add system prompt and format
            if system_prompt:
                prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                prompt = f"[INST] {prompt} [/INST]"
                
            body = {
                "prompt": prompt,
                **model_config.to_request_body()
            }
        else:
            # For other models (Mistral), concatenate context
            final_prompt = formatted_prompt
            if isinstance(formatted_prompt, list):
                final_prompt = "\n".join(msg["content"] for msg in formatted_prompt)
            
            body = {
                "prompt": final_prompt,
                **model_config.to_request_body()
            }
        
        # Make API call with retries
        client = get_bedrock_client(streaming=True)
        complete_response = ""
        
        for attempt in range(stream_config.retry_attempts):
            try:
                response = client.invoke_model_with_response_stream(
                    modelId=model_info.model_id,
                    body=json.dumps(body)
                )
                
                # Process stream
                print("\nResponse: ", end="")
                for chunk in process_stream_chunks(response, stream_config, print_fn):
                    print(chunk, end="", flush=True)
                    complete_response += chunk
                    
                print(f"\n\nRequest completed in {time.time() - start_time:.2f} seconds")
                
                # Update chat history if available
                if chat_history is not None:
                    chat_history.add_message("assistant", complete_response)
                
                return complete_response
                
            except ClientError as e:
                if "ThrottlingException" in str(e) and attempt < stream_config.retry_attempts - 1:
                    handle_rate_limit(attempt + 1, stream_config, print_fn)
                    continue
                raise
                
    except ClientError as e:
        error_msg = f"\n❌ AWS API error: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"\n❌ Error: {str(e)}"
        print(error_msg)
        raise
