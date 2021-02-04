import argparse
import logging
import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from sentiment import MODEL_VERSION


def upload_file(file_name, bucket):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('YOUR_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('YOUR_SECRET_KEY'))
    try:
        response = s3_client.upload_file(file_name,
                                         bucket,
                                         file_name,
                                         ExtraArgs={'ACL': 'public-read'})
    except ClientError as e:
        logging.error(e)
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default=f'model_{MODEL_VERSION}.pkl',
                        type=str)
    args = parser.parse_args()

    load_dotenv()
    upload_file(args.model_path, 'sentimentmodels')