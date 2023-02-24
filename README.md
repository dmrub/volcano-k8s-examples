# volcano-k8s-examples

This repository contains examples of using the [Volcano](https://volcano.sh/) scheduler to run pytorch and tensorflow jobs. There is also an example of using
[kubectl](https://kubernetes.io/docs/reference/kubectl/) to access Kubernetes within the pod.

**Note: all examples use the default volcano queue, if you want to use a different one you need to change the queue field in the files [volcano-gpu-pytorch-multi-worker.yaml](pytorch/volcano-gpu-pytorch-multi-worker.yaml) and [volcano-gpu-tf-keras-multi-worker.yaml](tensorflow/volcano-gpu-tf-keras-multi-worker.yaml).**

* [How to run pytorch jobs with Volcano](pytorch/README.md)
* [How to run a Tensorflow job with Volcano](tensorflow/README.md)
* [How to access Kubernetes within the pod](kubectl-shell/README.md)
