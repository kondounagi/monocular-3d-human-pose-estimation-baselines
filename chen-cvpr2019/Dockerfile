# ref. https://github.com/PyTorchLightning/pytorch-lightning/blob/master/dockers/base-cuda/Dockerfile
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    gcc \
    build-essential
RUN pip install --no-cache-dir \
    pylint \
    flake8 \
    black \
    pytorch-lighting \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
