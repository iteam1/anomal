from keras.models import load_model
import numpy as np
import cv2
import os
import random
import sys

# init
DIM = 256
src = 'test/noise/left'
dst = 'results'

# load model
model_border_dir = "model/unet/border/model.hdf5"
model_line_dir = "model/unet/lines/model.hdf5"
model_line = load_model(model_line_dir)
model_border = load_model(model_border_dir)

# read all images
images= os.listdir(src)
n = len(images)

# create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)

def mask(img,model_line,model_border):
    # Convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(gray, axis=0)
    x = x.astype('float32')
    
    # Find lines
    pred_line = model_line.predict([x])
    pred_line = pred_line.reshape(pred_line.shape[1],pred_line.shape[2])
    pred_line = pred_line * 255
    pred_line = pred_line.astype('uint8')
    
    # Find border
    pred_border = model_border.predict([x])
    pred_border = pred_border.reshape(pred_border.shape[1],pred_border.shape[2])
    pred_border = pred_border * 255
    pred_border = pred_border.astype('uint8')
    
    # Combine
    pred = pred_line + pred_border
    pred = cv2.bitwise_not(pred)
    
    out =  cv2.bitwise_and(img,img,mask = pred)
    return out

for i,image in enumerate(images):
    # read image
    print(i+1,"/",n,":",image)
    path = os.path.join(src,image)
    img = cv2.imread(path)
    # generate mask
    out = mask(img,model_line,model_border)
    #write out
    path = os.path.join(dst,image)
    cv2.imwrite(path,out)