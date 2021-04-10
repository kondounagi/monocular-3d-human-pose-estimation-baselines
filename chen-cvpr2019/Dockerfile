# ref. https://github.com/PyTorchLightning/pytorch-lightning/blob/master/dockers/base-cuda/Dockerfile
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    gcc \
    build-essential \
    && pip install --no-cache-dir \
    autopep8 \
    flake8 \
    black \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
