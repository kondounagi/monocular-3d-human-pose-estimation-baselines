#!/bin/bash

source params.sh
# docker run --rm -it ${GPU} ${NET} ${IPC} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash
docker run --rm -it ${GPU} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash
