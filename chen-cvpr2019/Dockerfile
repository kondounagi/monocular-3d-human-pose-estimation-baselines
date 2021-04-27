# ref. https://github.com/PyTorchLightning/pytorch-lightning/blob/master/dockers/base-cuda/Dockerfile
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.7

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    gcc \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6
RUN pip install --upgrade pip \
    && pip install --upgrade setuptools \
    && pip install --no-cache-dir \
    pylint \
    flake8 \
    black \
    pytorch-lightning \
    mlflow \
    opencv-contrib-python \
    optuna \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
