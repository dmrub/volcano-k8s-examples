#!/usr/bin/env python3

# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb
import json
import os
import sys

# In a real-world application, each worker would be on a different machine.
# For the purposes of this tutorial, all the workers will run on the this machine.
# Therefore, disable all GPUs to prevent errors caused by all workers trying to use the same GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Reset the TF_CONFIG environment variable
os.environ.pop("TF_CONFIG", None)

# Make sure that the current directory is on Python's path
if "." not in sys.path:
    sys.path.insert(0, ".")

import tensorflow as tf

import mnist_setup

# Single worker
batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
