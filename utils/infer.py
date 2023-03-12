import os
import sys
import cv2
import numpy as np
from anomalib.post_processing import ImageResult
from anomalib.deploy import TorchInferencer, OpenVINOInferencer

# model anomal
config_path = 'model/stfpm/mvtec/laptop/run/config.yaml'
model_path = 'model/stfpm/mvtec/laptop/run/weights/model.ckpt'
inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

# model border

# model lines

# read input image
img = cv2.imread('datasets/laptop/noise/000.png')

if __name__ == "__main__":

    prediction = inferencer.predict(image=img)
    pred_label = prediction.pred_label
    pred_score = prediction.pred_score
    print(pred_label,pred_score)
