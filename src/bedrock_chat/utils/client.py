"""AWS Bedrock client utilities."""

import os
import boto3
from typing import Optional, Union
from botocore.client import BaseClient
from dotenv import load_dotenv

def get_bedrock_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    streaming: bool = False
) -> BaseClient:
    """Get AWS Bedrock client.
    
    Args:
        aws_access_key_id: Optional AWS access key ID
        aws_secret_access_key: Optional AWS secret access key
        region_name: Optional AWS region name
        streaming: Whether to return a streaming-capable runtime client
        
    Returns:
        Configured boto3 Bedrock client
        
    Note:
        If credentials are not provided, they will be loaded from environment variables
        or AWS configuration files.
    """
    load_dotenv()
    
    service_name = 'bedrock-runtime'
    
    return boto3.client(
        service_name=service_name,
        aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=region_name or os.getenv('AWS_REGION')
    )
