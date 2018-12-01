import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

#role = get_execution_role()
role = 'arn:aws:iam::339215578670:role/service-role/AmazonSageMaker-ExecutionRole-20181126T120986'

###############################

#inputs = sagemaker_session.upload_data(path='data', key_prefix='data/DEMO-abalone')
inputs = 's3://sagemaker-eu-central-1-339215578670/data/DEMO-abalone'

###############################

# !cat 'abalone.py'

###############################

from sagemaker.tensorflow import TensorFlow

model_artifacts_location = 's3://sagemaker-eu-central-1-339215578670/artifacts'
checkpoint_path_location = 's3://sagemaker-eu-central-1-339215578670/checkpoints'

abalone_estimator = TensorFlow(entry_point='abalone.py',
                               output_path=model_artifacts_location,
                               checkpoint_path=checkpoint_path_location,
                               role=role,
                               framework_version='1.11.0',
                               training_steps= 200,                                  
                               evaluation_steps= 100,
                               hyperparameters={'learning_rate': 0.001},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge')

abalone_estimator.fit(inputs)

###############################

abalone_predictor = abalone_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

###############################

import tensorflow as tf
import numpy as np

prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=os.path.join('data/abalone_predict.csv'), target_dtype=np.int, features_dtype=np.float32)

data = prediction_set.data[0]
tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)

###############################

abalone_predictor.predict(tensor_proto)

###############################

sagemaker.Session().delete_endpoint(abalone_predictor.endpoint)



