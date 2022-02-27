FROM nvidia/cuda:10.0-devel

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
# install conda
# ------------------------------------------------------------------
RUN curl -o ~/anaconda.sh https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/anaconda.sh && \
    ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# ==================================================================
# install python
# ------------------------------------------------------------------
RUN conda install -y python=3.7 && \
    conda update --all

# ==================================================================
# install tensorflow, pytorch and utils
# ------------------------------------------------------------------
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

COPY . /root

WORKDIR /root