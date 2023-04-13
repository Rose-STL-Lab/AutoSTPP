from botocore.client import Config
from botocore import UNSIGNED
import boto3
import os
from loguru import logger
import argparse


def download(prefix="data/"):
    s3 = boto3.resource('s3', 
                        endpoint_url='https://s3-west.nrp-nautilus.io', 
                        config=Config(signature_version=UNSIGNED))
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
