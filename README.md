# PerSF-SemCom: Personalized Saliency-Based Semantic Communication

TODO: INTRODUCTION TO THIS WORK.

## Requirements

* nvidia-docker
* cuda>=10.0

## Quick Start

### Option 1 (Recommend): Download prebuilt docker image
The authors have encapsulated the development environment, all the 
required python packages, code and data into a docker image, which 
is open available on DockerHub, to help reproduce our experiment 
environment quickly. (Note that the image size is relatively large, 
it takes about 11.7GB of space on the disk.)

**1. Pull docker image from DockerHub:**
> $ sudo docker pull lizonghango00o1/persf-semcom:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0

**2. Run a container from the image:**
> $ sudo docker run -dit --name persf-semcom --gpus all lizonghango00o1/persf-semcom:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0

**3. Enter the container and run PerSF-SemCom:**
> $ sudo docker exec -ti persf-semcom bash \
> (persf-semcom) $ python -W ignore main.py \
> Personalized query text: {0: 'woman has hair', 1: 'sign on building', 2: 'woman wearing shirt'} \
> (Wait a few minutes for simulating the multipath fading channel ...) \
> [Bargain:schedule+alpha0.2] Best power allocation (ratio): [0.40640748 0.18427428 0.40010618] \
> [Bargain:schedule+alpha0.2] Best scores: [0.45763, 0.11525, 0.3322] \
> [Bargain:schedule+alpha0.2] Best target: 0.0175208450615 

Default setting: Total transmitter power 3kW, number of users is 3, 
fusion coefficient alpha=0.2. Multiprocessing is default enabled for
acceleration.

### Option 2: Build image from Dockerfile
For a more flexible and lightweight refactoring of our experiment 
environment, we provide a Dockerfile to help rebuild the docker image.

**1. Clone this Git repository.**

**2. Download the [pretrained RelTR weights](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) and put it under ``RelTR/ckpt/``.**

**3. Put custom data files under ``data/custom_data/data/``.**

**4. Build the docker image.**

This Dockerfile requires CUDA>=10.0 on the host machine, and it will
copy all files to the image.

> $ sudo docker build -f Dockerfile -t persf-semcom:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0 . \
> $ sudo docker run -dit --name persf-semcom --gpus all persf-semcom:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0 \
> (container) $ python -W ignore main.py --input_dir data/custom_data/data --output_dir data/custom_data/output --resume_pkl 0

This will load images from ``input_dir`` and save output to ``output_dir``, 
including the outputs of RelTR and Saliency in pickle format, and the 
visualization figures of AttnFusion. Set ``--resume_pkl=1`` to reuse 
preprocessed outputs.

You can also mount the data directory to the container, using the ``-v`` 
option, for example,

> $ sudo docker run -dit --name persf-semcom --gpus all -v {PATH_TO_PROJECT}/data:/root/data persf-semcom:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0

## Other options



# Acknowledgements
PerSF-SemCom uses [RelTR](https://github.com/yrcong/RelTR) and 
[Saliency](https://github.com/alexanderkroner/saliency) as attention
backends, and [TextMatch](https://github.com/MachineLP/TextMatch) as
score estimation, and [scikit-opt](https://github.com/guofei9987/scikit-opt)
as RCGA core. We sincerely thank them for their work that underpins this research.