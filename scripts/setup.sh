# for inference enviroment
#!usr/bin/bash

virtualenv env
source env/bin/activate

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install anomalib==0.4.0
#pip install openvino-dev==2021.3.0
#pip install -r requirements/base.txt
#pip install -r requirements/openvino.txt