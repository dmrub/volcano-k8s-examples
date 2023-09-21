#!/usr/bin/env bash

set -eou pipefail

error() {
    echo >&2 "* Error: $*"
}

fatal() {
    error "$@"
    exit 1
}

message() {
    echo >&2 "$*"
}

print-help() {
    echo "$(basename "$0") [<options>] NAMESPACE

Creates kubectl-access ServiceAccess object and RBAC rules for it in the specified namespace.
ServiceAccess kubectl-access will allow the Kubernetes pod to access all resources in the namespace.
By default, the generated objects are output to stdout.

options:
   -o,--output= FILENAME     Output generated RBAC rules to file FILENAME instead of stdout
      --help                 Display this help and exit
      --                     End of options

    "
}

OUTPUT=

while [[ $# -gt 0 ]]; do
    case "$1" in
    --help)
        print-help
        exit
        ;;
    -o | --output)
        OUTPUT="$2"
        shift 2
        ;;
    --output=*)
        OUTPUT="${1#*=}"
        shift
        ;;
    --)
        shift
        break
        ;;
    -*)
        fatal "Unknown option $1"
        ;;
    *)
        break
        ;;
    esac
done

if [[ $# -ne 1 ]]; then
    error "Single namespace argument required"
    print-help
    exit 1
fi

NS=$1

message "Set namespace to '$NS'"

print-rbac() {
    echo "---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kubectl-access
  namespace: ${NS}
---
kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  namespace: ${NS}
  name: cluster-ns-owner
rules:
- apiGroups: [\"\"]
  resources: [\"nodes\", \"namespaces\", \"secrets\", \"configmaps\"]
  verbs: [\"list\", \"get\", \"watch\"]
- apiGroups: [\"\", \"extensions\", \"apps\"]
  resources: [\"deployments\", \"replicasets\", \"pods\", \"persistentvolumeclaims\", \"services\", \"endpoints\", \"replicationcontrollers\", \"daemonsets\", \"statefulsets\", \"events\", \"secrets\", \"configmaps\", \"deployments/scale\"]
  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"patch\", \"delete\"]
- apiGroups: [\"autoscaling\"]
  resources: [\"horizontalpodautoscalers\"]
  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"patch\", \"delete\"]
- apiGroups: [\"\"]
  resources: [\"pods\", \"pods/log\"]
  verbs: [\"get\", \"list\", \"watch\"]
- apiGroups: [\"\"]
  resources: [\"pods/exec\", \"pods/portforward\", \"pods/attach\"]
  verbs: [\"create\"]
- apiGroups: [\"batch\"]
  resources: [\"jobs\", \"cronjobs\"]
  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"patch\", \"delete\"]
- apiGroups: [\"batch.volcano.sh\"]
  resources: [\"jobs\", \"cronjobs\"]
  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"patch\", \"delete\"]
- apiGroups: [\"metrics.k8s.io\"]
  resources: [\"pods\"]
  verbs: [\"get\", \"list\", \"watch\"]
- apiGroups: [\"kubeflow.org\"]
  resources: [\"poddefaults\"]
  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"patch\", \"delete\"]
---
kind: RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: cluster-ns-owner-rb
  namespace: ${NS}
subjects:
- kind: ServiceAccount
  name: kubectl-access
  namespace: ${NS}
roleRef:
  kind: Role
  name: cluster-ns-owner
  apiGroup: \"\"
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  annotations:
    rbac.authorization.kubernetes.io/autoupdate: \"true\"
  name: \"${NS}-cluster-ns-owner\"
rules:
- apiGroups: [\"\"]
  resources: [\"nodes\"]
  verbs: [\"get\", \"watch\", \"list\"]
- apiGroups: [\"metrics.k8s.io\"]
  resources: [\"nodes\"]
  verbs: [\"get\", \"list\", \"watch\"]
- apiGroups: [\"scheduling.volcano.sh\"]
  resources: [\"queues\"]
  verbs: [\"get\", \"list\", \"watch\"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: \"${NS}-cluster-ns-owner-crb\"
subjects:
- kind: ServiceAccount
  name: kubectl-access
  namespace: ${NS}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: \"${NS}-cluster-ns-owner\""
}

if [[ -n "$OUTPUT" && "$OUTPUT" != "-" ]]; then
    message "Output to file '$OUTPUT'"
    print-rbac >"$OUTPUT"
else
    print-rbac
fi
