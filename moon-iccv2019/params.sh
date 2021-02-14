#!/bin/bash

NAME=${PWD##*/}
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"
GPU="--gpus all"

VOLUMES="-v $(pwd):/workspace/"
