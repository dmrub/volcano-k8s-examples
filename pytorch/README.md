# pytorch example for Volcano

Source code based on:

* https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
* https://github.com/yangkky/distributed_tutorial

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
    kubectl apply -f pytorch-codeserver.yaml
    ```
5. Connect to the running pytorch-codeserver with port-forward
    ```sh
    kubectl port-forward pytorch-codeserver-5989bb6b8b-44spg 8888
    ```
    The name of the pytorch code server pod is different every time you start it, so you have to use `kubectl get pods` to find out the correct name.
6. In the codeserver use `git clone` to clone this repository:
    ```sh
    git clone https://github.com/dmrub/volcano-k8s-examples.git
    ```
7. Run the pytorch multiworker job
    ```sh
    kubectl apply -f volcano-gpu-pytorch-multi-worker.yaml
    ```
8. Use `kubectl get pods` to check the current status and `kubectl logs POD_NAME` to check the output of one of the running pods. You can use the code server to edit the source code between starts of your job. You can log into the running container with the command `kubectl exec -ti POD_NAME -- /bin/bash`.