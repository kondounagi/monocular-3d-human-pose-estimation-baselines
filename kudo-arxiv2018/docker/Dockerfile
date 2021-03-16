FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Ubuntu dependencies.
RUN apt-get update -qq \
    && apt-get install --no-install-recommends -y \
    sudo git curl grep sed dpkg wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 libgl1-mesa-dev \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/log/dpkg.log

# Install Miniconda.
ENV PATH /opt/conda/bin:$PATH
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda install -y python=3.8 \
    && conda update conda \
    && conda clean --all --yes

# Install Python dependencies.
COPY requirements.txt /work/requirements.txt
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda create -n cvdev python=3.8 -y \
    && conda activate cvdev \
    && pip install pip==21.0.1 boto3==1.17.15 awscli==1.19.15 \ 
    && pip install -r /work/requirements.txt \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate cvdev" >> ~/.bashrc \
    && rm -rf /root/.cache/pip/*

WORKDIR /work
