import os

import boto3

print(os.environ)
session = boto3.Session(region_name='us-east-1')# -*- coding: utf-8 -*-

for bucket in boto3.resource('s3').buckets.all():
    print(bucket.name)

import sagemaker
