import boto3
from botocore.config import Config
import logging
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Long timeout for reasoning/thinking models (up to 60 min per AWS Nova docs)
_BEDROCK_CONFIG = Config(read_timeout=3600, connect_timeout=60, retries={"max_attempts": 2})

@lru_cache(maxsize=16)
def get_aws_client(service_name, region=None, endpoint_url=None):
    """
    Get a cached boto3 client to avoid creating new connections
    """
    logger.debug(f"Getting AWS client for {service_name} in region {region}")
    kwargs = {}
    if region:
        kwargs['region_name'] = region
    if endpoint_url:
        kwargs['endpoint_url'] = endpoint_url
    if service_name in ('bedrock-runtime', 'bedrock'):
        kwargs['config'] = _BEDROCK_CONFIG
        
    return boto3.client(service_name, **kwargs)

@lru_cache(maxsize=8)
def get_aws_session(region=None):
    """
    Get a cached boto3 session
    
    Args:
        region: AWS region name
        
    Returns:
        Boto3 session
    """
    if region:
        return boto3.session.Session(region_name=region)
    return boto3.session.Session()

def get_account_id():
    """
    Get the current AWS account ID
    
    Returns:
        AWS account ID
    """
    sts_client = get_aws_client('sts')
    return sts_client.get_caller_identity()['Account']

def get_current_region():
    """
    Get the current AWS region
    
    Returns:
        AWS region name
    """
    return get_aws_session().region_name