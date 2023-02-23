#!/usr/bin/env bash

set -eou pipefail

THIS_DIR=$( cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P )

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
    echo "$(basename "$0") [--help] NAMESPACE"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            print-help
            exit
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

cd "$THIS_DIR"
#set -x
for FN in *.yaml; do
    message "Processing $FN"
    KIND=$(grep "kind:" "$FN" | sed -En 's/^[[:space:]]*kind:[[:space:]]*([[:alnum:]]*)[[:space:]]*$/\1/gp')
    #echo "$KIND"
    if [[ "$KIND" = "Namespace" ]]; then
        printf -v SED_EXPR 's/^([[:space:]]*)name:([[:space:]]*)[[:alnum:]]*([[:space:]]*)$/\\1name: %s/g' "$NS"
    else
        printf -v SED_EXPR 's/^([[:space:]]*)namespace:([[:space:]]*)[[:alnum:]]*([[:space:]]*)$/\\1namespace: %s/g' "$NS"
    fi
    sed -E "$SED_EXPR" "$FN" | \
        sed -E "s/system:serviceaccount:([[:alnum:]]+):default/system:serviceaccount:${NS}:default/g" | \
        sed -E "s/([[:alnum:]]+)-cluster-user/${NS}-cluster-user/g" | \
        sed -E "s/([[:alnum:]]+)-kubectl-access/${NS}-kubectl-access/g" | \
        sed -E "s/([[:alnum:]]+)-cluster-ns-owner/${NS}-cluster-ns-owner/g"  > "${FN}.new"
    mv "${FN}.new" "$FN"
done
message "Done"
