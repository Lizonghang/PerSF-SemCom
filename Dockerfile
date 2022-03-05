FROM lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0

# ==================================================================
# apt tools
# ------------------------------------------------------------------
RUN APT_INSTALL="apt install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        zip \
        unzip \
        vim \
        wget \
        curl \
        git \
        aria2 \
        apt-transport-https \
        openssh-client \
        openssh-server \
        libopencv-dev \
        libsnappy-dev \
        tzdata \
        iputils-ping \
        net-tools \
        htop

# ==================================================================
# install conda (NOTE: Already installed in the base image, python 3.6.8)
# ------------------------------------------------------------------
RUN curl -o /root/anaconda.sh https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /root/anaconda.sh && \
    /root/anaconda.sh -bu -p /opt/conda && \
    rm /root/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# ==================================================================
# install python
# ------------------------------------------------------------------
RUN conda install -y python=3.7 && \
    conda update --all

# ==================================================================
# install tensorflow, pytorch and utils
# ------------------------------------------------------------------
RUN pip install scikit-learn==0.21.3 \
                jieba==0.42.1 \
                easydict==1.9

RUN pip install torch==1.6.0 \
                torchvision==0.7.0

RUN pip install tensorflow==1.13.1 \
#                tensorflow-gpu==1.13.1 \
                matplotlib==3.0.3 \
                requests==2.21.0 \
                scikit-image==0.14.2 \
                scipy==1.4.1 \
                numpy==1.16.4 && \
    yes | pip uninstall h5py && \
    pip install --no-cache-dir h5py

# NOTE: NEED TO INSTALL cudnn-10.0-linux-x64-v7.4.2.24.tgz
#   MANUALLY IF TENSORFLOW-GPU IS REQUIRED.

USER root

WORKDIR /root