#!/bin/bash
cd $(dirname $0)

NAME="chen-cvpr2019"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
GPU="--gpus all"

VOLUMES="-v $(pwd)/src:/workspace/ -v $(pwd)/data:/data/"
