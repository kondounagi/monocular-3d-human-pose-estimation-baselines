#!/bin/bash
cd $(dirname $0)

NAME="chen-cvpr2019"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"
GPU="--gpus all"

VOLUMES="-v $(pwd)/src:/workspace/ -v $(pwd)/data:/data/"
