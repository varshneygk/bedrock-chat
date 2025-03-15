"""Streaming utilities for AWS Bedrock Chat."""

import json
from typing import Dict, Any, Generator, Optional, Callable
from ..models.model_config import StreamConfig

def process_stream_chunks(
    response: Dict[str, Any],
    config: StreamConfig,
    print_fn: Optional[Callable[[str], None]] = print
) -> Generator[str, None, None]:
    """Process streaming response chunks.
    
    Args:
        response: Bedrock streaming response
        config: Stream configuration
        print_fn: Optional function to print status messages
        
    Yields:
        Text content from each chunk
    """
    chunk_count = 0
    accumulated_text = ""
    
    if print_fn:
        print_fn("\nProcessing stream...")
    
    for event in response.get('body', []):
        chunk = json.loads(event.get('chunk', {}).get('bytes', b'{}').decode())
        chunk_count += 1
        
        # Handle different model response formats
        if 'completion' in chunk:  # Claude 2 format
            text = chunk['completion']
            accumulated_text += text
            yield text
        elif 'type' in chunk:  # Claude 3 streaming format
            if chunk['type'] == 'content_block_delta':
                text = chunk.get('delta', {}).get('text', '')
                accumulated_text += text
                yield text
            elif chunk['type'] == 'message_delta':
                text = chunk.get('delta', {}).get('content', [{}])[0].get('text', '')
                accumulated_text += text
                yield text
        elif 'outputText' in chunk:  # Titan format
            text = chunk['outputText']
            accumulated_text += text
            yield text
        elif 'generation' in chunk:  # Llama format
            text = chunk['generation']
            accumulated_text += text
            yield text
        elif 'text' in chunk:  # Mistral format
            text = chunk['text']
            accumulated_text += text
            yield text
        elif 'outputs' in chunk:  # Generic format
            for output in chunk['outputs']:
                text = output.get('text', '')
                accumulated_text += text
                yield text
                
        if print_fn and chunk_count % 5 == 0:
            print_fn(f"Processed chunks: {chunk_count}")
    
    if print_fn:
        print_fn(f"\n✅ Successfully streamed {len(accumulated_text)} characters in {chunk_count} chunks\n")
