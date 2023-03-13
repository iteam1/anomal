import os
import sys
import cv2
import pickle
import numpy as np
from keras.models import load_model
from anomalib.deploy import TorchInferencer

# init
#src = 'test/crack0'
src = 'test/noise3'
#src = 'test/good0'
side =  'left'
src = os.path.join(src,side)
dst = 'results'
THRESH = 0.5
DIM = 256
s = 3
k = 124
T = 500
v = 1
count = 0 # count anomalous

# model anomal
config_path = 'model/stfpm/mvtec/laptop/run/config.yaml'
model_path = 'model/stfpm/mvtec/laptop/run/weights/model.ckpt'
inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

if __name__ == "__main__":
    # list all images
    images = os.listdir(src)
    n = len(images)
    # predict
    for i,image in enumerate(images):
        print(i+1,"/",n,":",image)
        path = os.path.join(src,image)
        # read input image
        img = cv2.imread(path)

        # first predict
        prediction = inferencer.predict(image=img)
        pred_label = prediction.pred_label
        pred_score = prediction.pred_score
        # retest
        if pred_label == 'Anomalous' and pred_score > THRESH:
            pred_mask = prediction.pred_mask # get mask
            path = os.path.join(dst,image)
            if side == 'left':
                corner = img[:k,:k]
                corner_mask = pred_mask[:k,:k]
                total = np.sum(corner_mask)/255
                # check total area
                if total > T:
                    count +=1
                    cv2.imwrite(path,corner)
            else:
                corner = img[:k,DIM-k:DIM]
                corner_mask = pred_mask[:k,DIM-k:DIM]
                total = np.sum(corner_mask)/255
                # check total area
                if total > T:
                    count +=1
                    cv2.imwrite(path,corner)

    print('Total anomalous:',count)
