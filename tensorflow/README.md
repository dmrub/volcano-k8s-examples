# Tensorflow example for Volcano

Source code based on:

* https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb

# Instructions

1. If necessary, replace the namespace field in the yaml files with your own or delete it if you want to use your default namespace.
2. Replace the name of the PVC in the YAML files if necessary, e.g. if you want to use your own PVC.
3. You can create a namespace either with `kubectl apply -f ns.yaml` or with `kubectl create namespace NAMESPACE_NAME`.
4. If you do not have your own PVC, create a PVC from the file `pvc.yaml` with the command:
    ```sh
    kubectl apply -f pvc.yaml
    ```
4. Start the codeserver to check out the code in the container using git:
    ```sh
    kubectl apply -f tf-codeserver.yaml
    ```
5. Connect to the running tf-codeserver with port-forward
    ```sh
    kubectl port-forward tf-codeserver-7cbbcb49c5-gwnvm 8888
    ```
    The name of the tensorflow code server pod is different every time you start it, so you have to use `kubectl get pods` to find out the correct name.
6. In the codeserver use `git clone` to clone this repository:
    ```sh
    git clone https://github.com/dmrub/volcano-k8s-examples.git
    ```
7. Run the tensorflow multiworker job
    ```sh
    kubectl apply -f volcano-gpu-tf-keras-multi-worker.yaml
    ```
8. Use `kubectl get pods` to check the current status and `kubectl logs POD_NAME` to check the output of one of the running pods. You can use the code server to edit the source code between starts of your job. You can log into the running container with the command `kubectl exec -ti POD_NAME -- /bin/bash`.
