# Project: Semantics Saliency

This project is a combination of [RelTR](https://github.com/yrcong/RelTR) and [Saliency](https://github.com/alexanderkroner/saliency).

## Requirements

* nvidia-docker
* cuda>=1.0

## Build Environment Via Docker

### Option 1: Manually Build Docker Image
First, clone this Git repository, manually download the [pretrained RelTR weights](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) and put it under ``RelTR/ckpt/``.
Then, the docker image can be built by:

> $ sudo docker build -f Dockerfile -t semsal .
>
> $ sudo docker run -it --rm --gpus all semsal bash
> 
> (container) $ python main.py

The process loads images from ``data/`` and saves output images and logs at ``output/``.

### Option 2: Use Pre-built Docker Image
Simply pull the pre-built docker image and run a container:

> $ sudo docker pull lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0
> 
> $ sudo docker run -it --rm --gpus all lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0 bash
> 
> (container) $ python main.py

NOTE: You can use your own input images and synchronize output files to the host machine by mounting the host path to the container path:

> $ sudo docker run -it --rm --gpus all -v /path/to/SemSal/data:/root/data -v /path/to/SemSal/output:/root/output lizonghango00o1/semsal:tf-cpu1.13.1-torch-gpu1.6.0-cuda10.0 bash

