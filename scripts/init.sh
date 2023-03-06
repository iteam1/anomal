# for training enviroment
#!usr/bin/bash

MYCONDA="myconda"
MYENV="myenv"

bash utils/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh

#$MYCONDA/bin/conda create -n $MYENV python=3.8
myconda/bin/conda create -n myenv python=3.8

myconda/envs/myenv/bin/pip list

myconda/envs/myenv/bin/pip install -r requirements/base.txt
myconda/envs/myenv/bin/pip install -r requirements/openvino.txt
myconda/envs/myenv/bin/pip install anomalib==0.4.0
myconda/envs/myenv/bin/pip install wandb