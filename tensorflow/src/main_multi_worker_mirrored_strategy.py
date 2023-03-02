#!/usr/bin/env python3

# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb
import os
import json
import shutil
import tensorflow as tf
import mnist_setup

if 'MLFLOW_TRACKING_URI' in os.environ:
    try:
        import mlflow
        import mlflow.keras
        print(f'INFO: Use mlflow with MLFLOW_TRACKING_URI = {os.environ["MLFLOW_TRACKING_URI"]}')
        mlflow.keras.autolog()
    except ImportError as e:
        print("WARNING: mlflow could not be imported, please install it first", file=sys.stderr)

per_worker_batch_size = 64
epochs = 3
# per_worker_batch_size = 128
# epochs = 35
tf_config = json.loads(os.environ["TF_CONFIG"])
num_workers = len(tf_config["cluster"]["worker"])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=epochs, steps_per_epoch=70)

# Save model

model_path = os.getenv("MODEL_PATH", None)
if model_path is None:
    print(
        "No model path specified in the environment variable MODEL_PATH, model will not be saved"
    )

temp_model_path = os.getenv("TEMP_MODEL_PATH", None)
if temp_model_path is None:
    print(
        "No temporary model path specified in the environment variable TEMP_MODEL_PATH, temporary model will not be saved"
    )

delete_temp_model_path = os.getenv("DELETE_TEMP_MODEL_PATH", "1")

delete_temp_model_path = delete_temp_model_path.lower() in ("1", "true", "yes")

def _is_chief(task_type, task_id):
    # Note: there are two possible `TF_CONFIG` configurations.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this Colab section, the `task_type` is `None` case
    # is added because it is effectively run with only a single worker.
    return (task_type == "worker" and task_id == 0) or task_type is None


def _get_temp_dir(dirpath, task_id):
    base_dirpath = "workertemp_" + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, temp_filepath, task_type, task_id):
    if _is_chief(task_type, task_id):
        if filepath is None:
            return None
        result = filepath
    else:
        if temp_filepath is None:
            return None
        result = os.path.join(temp_filepath, "workertemp_" + str(task_id))
    os.makedirs(result, exist_ok=True)
    return result


task_type, task_id = (
    strategy.cluster_resolver.task_type,
    strategy.cluster_resolver.task_id,
)
print("task_type = {}".format(task_type))
print("task_id = {}".format(task_id))

write_model_path = write_filepath(model_path, temp_model_path, task_type, task_id)
if write_model_path is not None:
    # Disable GC as a workaround for https://github.com/tensorflow/tensorflow/issues/50853
    import gc
    gc.disable()
    print("Save model to {}".format(write_model_path))
    multi_worker_model.save(write_model_path)
    #print("Save weights to {}".format(write_model_path))
    #multi_worker_model.save_weights(write_model_path)

    if not _is_chief(task_type, task_id) and delete_temp_model_path:
        print("Delete tree {}".format(write_model_path))
        tf.io.gfile.rmtree(write_model_path)
        # shutil.rmtree(write_model_path, ignore_errors=False)

    print("Save done")
    # gc.enable()
print("END")
