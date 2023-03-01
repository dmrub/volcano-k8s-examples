# Tensorflow example for Volcano

Source code based on:

* https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb

# Instructions

1. If necessary, replace the namespace field in the yaml files with your own or delete it if you want to use your default namespace. You can use for this following shell command in this directory or use script `set-namespace.sh`:
    ```sh
    ./set-namespace.sh NEW_NAMESPACE_NAME
    # Or
    for f in *.yaml; do echo "$f"; sed -i "s/dmrub/NEW_NAMESPACE_NAME/g" "$f"; done
    ```
2. Replace the name of the PVC in the YAML files if necessary, e.g. if you want to use your own PVC.
3. You can create a namespace either with `kubectl apply -f ns.yaml` or with `kubectl create namespace NAMESPACE_NAME`.
4. If you do not have your own PVC, create a PVC from the file `pvc.yaml` with the command:
    ```sh
    kubectl apply -f pvc.yaml
    ```
5. Start the codeserver to check out the code in the container using git:
    ```sh
    kubectl apply -f tf-codeserver.yaml
    ```
6. Connect to the running tf-codeserver with port-forward
    ```sh
    kubectl port-forward tf-codeserver-7cbbcb49c5-gwnvm 8888
    ```
    The name of the tensorflow code server pod is different every time you start it, so you have to use `kubectl get pods` to find out the correct name.
7. In the codeserver use `git clone` to clone this repository:
    ```sh
    git clone https://github.com/dmrub/volcano-k8s-examples.git
    ```
    Alternatively, you can initialise the source code with a Kubernetes job that clones the repository and then exits:
    ```sh
    kubectl apply -f init-job.yaml
    ```
8. Run the tensorflow job with MirroredStrategy or with the MultiWorkerMirroredStrategy
    ```sh
    kubectl apply -f volcano-gpu-tf-keras-mirrored-strategy.yaml
    # Or
    kubectl apply -f volcano-gpu-tf-keras-multi-worker-mirrored-strategy.yaml
    ```
9. Use `kubectl get pods` to check the current status and `kubectl logs POD_NAME` to check the output of one of the running pods. You can use the code server to edit the source code between starts of your job. You can log into the running container with the command `kubectl exec -ti POD_NAME -- /bin/bash`.
10. To access Kubernetes from the code server IDE, the Kubernetes administrator must apply the `rbac.yaml` file. Note that `rbac.yaml` grants permission to the default security account of the namespace used by all pods. If you want to grant Kubernetes access only to a subset of pods, you must follow the instructions in the `kubectl-shell` directory in the root of this repository [README.md](../kubectl-shell/README.md). In addition, you must add the following lines to your `~/.bashrc` file in the code server IDE:
    ```sh
    source /usr/share/bash-completion/bash_completion
    source <(kubectl completion bash | sed 's/--request-timeout=5s/--request-timeout=0/g')
    ```
    You can use the command `code-server ~/.bashrc` to open the file `.bashrc` for editing in the IDE.
