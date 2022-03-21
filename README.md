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
fusion coefficient $\alpha=0.2$.

### Option 2: Build image from Dockerfile
For a more flexible and lightweight refactoring of our experiment 
environment, we provide a Dockerfile to help rebuild the docker image.


First, clone this Git repository, manually download the [pretrained RelTR weights](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) and put it under ``RelTR/ckpt/``.
Then, the docker image can be built by:

> $ sudo docker build -f Dockerfile -t semsal .
>
> $ sudo docker run -it --rm --gpus all semsal bash
> 
> (container) $ python -W ignore main.py

The process loads images from ``data/`` and saves output images and logs at ``output/``.


Simply pull the pre-built docker image and run a container:

> $ sudo docker pull lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0
> 
> $ sudo docker run -it --rm --gpus all lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0 bash
> 
> (container) $ python -W ignore main.py

NOTE: You can use your own input images and synchronize output files to the host machine by mounting the host path to the container path:

> $ sudo docker run -it --rm --gpus all -v /path/to/SemSal/data:/root/data -v /path/to/SemSal/output:/root/output lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0 bash

# Acknowledgements
This project is a combination of [RelTR](https://github.com/yrcong/RelTR), [Saliency](https://github.com/alexanderkroner/saliency), and [TextMatch](https://github.com/MachineLP/TextMatch).