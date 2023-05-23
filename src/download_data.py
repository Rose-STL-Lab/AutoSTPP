import boto3
import os
from loguru import logger
import argparse


def download(prefix="data/"):
    s3 = boto3.resource('s3',
                        endpoint_url='https://f301160e44a0ed1c9e6a9cd6be3690f5.r2.cloudflarestorage.com',
                        aws_access_key_id='8c5b3e69fd3dce39a320ee95d0e7e0c7',
                        aws_secret_access_key='c0a5d9420d524580e9817e7c596265947f40b05c13d42abdf2b7fb6d13f312b0',
                        region_name='auto')
    bucket = s3.Bucket('autoint')

    for obj in bucket.objects.all():
        key = obj.key
        if key.startswith(prefix):
            name = key.split("/")[-1]
            local_path = os.path.dirname(key)
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            logger.info(f"Downloading {os.path.join(local_path, name)}...")
            bucket.download_file(key, os.path.join(local_path, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', help='Prefix for S3 file path', default="data/")
    args = parser.parse_args()
    download(args.prefix)
