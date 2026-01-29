from typing import Dict
from urllib.parse import quote
import asyncio
import functools

from botocore.client import Config

from app.api.infra.aws import session
from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.entity.object import S3ListObject, S3Object


async def get_object(key: str, bucket_name: str) -> S3Object:
    loop = asyncio.get_event_loop()
    client = session.client("s3")
    
    def _get():
        return client.get_object(Bucket=bucket_name, Key=key)
        
    object_ = await loop.run_in_executor(None, _get)
    return S3Object(**object_)


async def list_object(prefix: str, bucket_name: str) -> S3ListObject:
    loop = asyncio.get_event_loop()
    client = session.client("s3")
    
    def _list():
        return client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
    list_ = await loop.run_in_executor(None, _list)
    return S3ListObject(**list_)


async def put_object(obj: S3Object, bucket_name: str) -> Dict:
    loop = asyncio.get_event_loop()
    client = session.client("s3")
    
    def _put():
        return client.put_object(
            Bucket=bucket_name, 
            Body=obj.body, 
            ContentType=obj.content_type, 
            Key=obj.key
        )
        
    object_ = await loop.run_in_executor(None, _put)
    return object_


async def delete_object(key: str, bucket_name: str) -> Dict:
    loop = asyncio.get_event_loop()
    client = session.client("s3")
    
    def _delete():
        return client.delete_object(Bucket=bucket_name, Key=key)
        
    object_ = await loop.run_in_executor(None, _delete)
    return object_


def generate_presigned_get_url(key: str, bucket_name: str, expires_in: int = 120, is_preview: bool = True):
    """Generate a presigned URL for getting an S3 object
    
    Args:
        key (str): S3 object key
        bucket_name (str): S3 bucket name
        expires_in (int, optional): URL expiration time in seconds. Defaults to 120.
        is_preview (bool, optional): If True, sets headers for preview. If False, sets for download. Defaults to True.
        
    Returns:
        str: Presigned URL
    """
    client = session.client("s3")
    
    # Set content type for PDF
    content_type = 'application/pdf'
    
    # Get filename from key and encode it for headers
    filename = key.split('/')[-1]
    encoded_filename = quote(filename)
    
    # Prepare parameters
    params = {
        "Bucket": bucket_name,
        "Key": key,
        "ResponseContentType": content_type,
    }
    
    # Set content disposition and headers
    if is_preview:
        params["ResponseContentDisposition"] = f'inline; filename="{encoded_filename}"'
        # Add headers to improve PDF preview in browser
        params.update({
            "ResponseCacheControl": "no-cache",
            "ResponseExpires": "0",
            "ResponseContentEncoding": "identity"
        })
    else:
        params["ResponseContentDisposition"] = f'attachment; filename="{encoded_filename}"'
    
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires_in,
        HttpMethod="GET")


def generate_presigned_put_url(key: str, time:int):
    client = session.client("s3", config=Config(signature_version="s3v4"))
    return client.generate_presigned_url(
        ClientMethod="put_object",
        Params={
            "Bucket": s3_bucket,
            "Key": key
        },
        ExpiresIn=time,
        HttpMethod="PUT")


async def copy_object(
    source_key: str,
    destination_key: str,
    bucket_name: str,
    content_disposition: str = None,
    content_type: str = None,
    metadata: Dict = None
) -> Dict:
    """Copy an S3 object to a new key within the same bucket
    
    Args:
        source_key (str): Source object key
        destination_key (str): Destination object key
        bucket_name (str): S3 bucket name
        content_disposition (str)
        content_type (str)
        metadata (Dict)
    Returns:
        Dict: Response from S3 copy_object operation
    """
    loop = asyncio.get_event_loop()
    client = session.client("s3")
    
    copy_source = {
        'Bucket': bucket_name,
        'Key': source_key
    }

    extra_args = {
        'CopySource': copy_source,
        'Bucket': bucket_name,
        'Key': destination_key
    }

    if metadata is not None:
        extra_args['Metadata'] = metadata
        extra_args['MetadataDirective'] = 'REPLACE'

    if content_type is not None:
        extra_args['ContentType'] = content_type

    if content_disposition is not None:
        extra_args['ContentDisposition'] = content_disposition

    def _copy():
        return client.copy_object(**extra_args)
        
    return await loop.run_in_executor(None, _copy)


async def head_object(key: str, bucket_name: str) -> Dict:
    loop = asyncio.get_event_loop()
    client = session.client("s3")
    
    def _head():
        return client.head_object(Bucket=bucket_name, Key=key)
        
    return await loop.run_in_executor(None, _head)

