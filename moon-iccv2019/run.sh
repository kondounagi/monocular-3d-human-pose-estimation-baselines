#!/bin/bash

source params.sh
docker run --rm -it ${GPU} ${NET} ${IPC} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash
