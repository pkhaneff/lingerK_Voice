import boto3
from app.core.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

session = boto3.session.Session(
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)