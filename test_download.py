# import boto3
# import os
# from dotenv import load_dotenv

# # Load .env
# load_dotenv()

# print("=== Debug Credentials ===")
# print(f"AWS_ACCESS_KEY_ID from env: {os.getenv('AWS_ACCESS_KEY', 'NOT SET')}")
# print(f"AWS_SECRET_ACCESS_KEY from env: {os.getenv('AWS_SECRET_KEY', 'NOT SET')[:10] if os.getenv('AWS_SECRET_KEY') else 'NOT SET'}...")

# # Check boto3 credentials
# session = boto3.Session()
# credentials = session.get_credentials()
# if credentials:
#     print(f"Boto3 Access Key: {credentials.access_key}")
#     print(f"Boto3 Secret Key: {credentials.secret_key[:10]}...")
# else:
#     print("NO CREDENTIALS!")
# print("========================\n")

# s3 = boto3.client('s3')

import boto3
import os
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

print("=== Debug Credentials ===")
access_key = os.getenv('AWS_ACCESS_KEY')
secret_key = os.getenv('AWS_SECRET_KEY')

print(f"AWS_ACCESS_KEY_ID from env: {access_key}")
print(f"AWS_SECRET_ACCESS_KEY from env: {secret_key[:10] if secret_key else 'NONE'}...")
print("========================\n")

# FORCE boto3 dùng credentials từ .env (QUAN TRỌNG!)
s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='ap-northeast-1'
)

# Test 1: GetBucketLocation
try:
    response = s3.get_bucket_location(Bucket='linger-voice-bucket')
    print(f"✓ GetBucketLocation OK: {response}")
except Exception as e:
    print(f"✗ GetBucketLocation FAILED: {e}")

# Test 2: Simple upload
try:
    test_data = b"Hello S3!" * 1000
    s3.put_object(
        Bucket='linger-voice-bucket',
        Key='test-simple.txt',
        Body=test_data
    )
    print("✓ Simple PutObject OK")
except Exception as e:
    print(f"✗ Simple PutObject FAILED: {e}")

# Test 3: Multipart upload
try:
    response = s3.create_multipart_upload(
        Bucket='linger-voice-bucket',
        Key='test-multipart.txt'
    )
    upload_id = response['UploadId']
    print(f"✓ CreateMultipartUpload OK: {upload_id}")
    
    # Abort
    s3.abort_multipart_upload(
        Bucket='linger-voice-bucket',
        Key='test-multipart.txt',
        UploadId=upload_id
    )
    print("✓ AbortMultipartUpload OK")
except Exception as e:
    print(f"✗ CreateMultipartUpload FAILED: {e}")