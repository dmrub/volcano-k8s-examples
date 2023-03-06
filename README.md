# volcano-k8s-examples

This repository contains examples of using the [Volcano](https://volcano.sh/) scheduler to run pytorch and tensorflow jobs. There is also an example of using
[kubectl](https://kubernetes.io/docs/reference/kubectl/) to access Kubernetes within the pod.

**Note: all examples use the default volcano queue, if you want to use a different one you need to change the queue field in the files [volcano-gpu-pytorch-multi-worker.yaml](pytorch/volcano-gpu-pytorch-multi-worker.yaml) and [volcano-gpu-tf-keras-multi-worker.yaml](tensorflow/volcano-gpu-tf-keras-multi-worker.yaml).**

* [How to run pytorch jobs with Volcano](pytorch/README.md)
* [How to run a Tensorflow job with Volcano](tensorflow/README.md)
* [How to access Kubernetes within the pod](kubectl-shell/README.md)

## Contents of a Volcano yaml file

The yaml files used in the examples are VolcanoJob descriptions, which are described in detail in the official Volcano documentation: https://volcano.sh/en/docs/vcjob/. The examples provided in this repository can be used as templates, and this section briefly describes the main fields used.

1. `spec.metadata.namespace` - Kubernetes namespace, usually assigned to you by the administrator, or you can create your own, https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
2. `spec.metadata.name` - name of the job, must be unique in your namespace
3. `spec.minAvailable` - the minimum number of running pods required to run the job should usually be equal to the number of replicas
4. `spec.tasks[0].replicas` - number of pods to run, if you want to run your code on a single host with multiple GPUs, use 1 in this field and specify the number of GPUs in the `spec.tasks[0].template.spec.containers[0].resources.limits.nvidia.com/gpu`, if you want to run your code on multiple nodes with multiple GPUs, use a number higher than 1 here, in this case you will probably use the network to collect the results of individual running pods.
5. `spec.tasks[0].template.spec.containers[0].image` - container image that you want to use for your run. You can use any Docker containers that are available in the public registry like **Docker Hub** or **GitHub Packages**. If you need to use a private registry, see this article on how to authorize Kubernetes to download the image from there: https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/
6. `spec.tasks[0].template.spec.containers[0].command` - list with the command to be executed in the container and its arguments. This command overrides the default command specified in the container and is optional. Typically, it is used to execute custom shell commands by using a shell interpreter, such as Bash, as the command and specifying shell commands in the arguments. The command, e.g. the shell interpreter, must already be installed in the container!
7. `spec.tasks[0].template.spec.containers[0].resources.limits.nvidia.com/gpu` - number of GPUs allocated for a single Pod. Note that the job remains pending until there are enough resources to execute it.
8. `spec.tasks[0].template.spec.containers[0].workingDir` - working directory where the command is executed
9. `spec.tasks[0].template.spec.containers[0].volumeMounts` - volumes to be mounted in the running container: https://kubernetes.io/docs/concepts/storage/volumes/
10. `spec.tasks[0].template.spec.containers[0].volumes` - specification of the mounted volumes: https://kubernetes.io/docs/concepts/storage/volumes/

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  namespace: dmrub # 1. Your namespace
  name: tf-keras-mirrored-strategy # 2. Name of your job
spec:
  minAvailable: 1 # 3. The minimum number of running pods required to run the job should usually be equal to the number of replicas
  schedulerName: volcano
  queue: default
  plugins:
    env: []
    svc: []
  policies:
    - event: PodEvicted
      action: RestartJob
  tasks:
    - replicas: 1 # 4. Number of pods to run
      name: worker
      policies:
        - event: TaskCompleted
          action: CompleteJob
      template:
        spec:
          imagePullSecrets:
            - name: default-secret
          containers:
            - name: tensorflow
              image: tensorflow/tensorflow:2.9.1-gpu # 5. Container image
              imagePullPolicy: Always
              command:
                # 6. Commands to be executed in the container
                - /bin/bash
                - -c
                - |
                  # Shell commands to be executed in the container's bash shell
                  set -xe;
                  # Output information about used GPU
                  nvidia-smi;
                  nvidia-smi --format=csv,noheader --query-gpu=name,serial,uuid,pci.bus_id;
                  # Run ML training program
                  python3 main_mirrored_strategy.py;
              ports:
                - containerPort: 2222
                  name: tfjob-port
              resources:
                limits:
                  nvidia.com/gpu: "3" # 7. Number of GPUs per pod
              workingDir: /data/volcano-k8s-examples/tensorflow/src # 8. Working directory
              # 9. Mounted volumes
              volumeMounts:
                - mountPath: /data
                  name: data-volume
          volumes:
            # Specification of the mounted volumes
            - name: data-volume
              persistentVolumeClaim:
                claimName: volcano-workspace
          restartPolicy: OnFailure
```
