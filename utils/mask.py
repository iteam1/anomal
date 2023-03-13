from keras.models import load_model
import numpy as np
import cv2
import os
import sys
from skimage.morphology import closing
from skimage.morphology import disk  # noqa

# init
footprint = disk(6)
kernel = np.ones((2, 2), np.uint8)
s = 3 # window side
p = 50 # patient value
DIM = 256
side = "left"
#src = 'test/crack'
src = 'test/noise'
#src = 'test/good'
src = os.path.join(src,side)
dst = 'results'

# load model
model_line_dir = "model/unet/lines/model.hdf5"
model_line = load_model(model_line_dir)

# read all images
images= os.listdir(src)
n = len(images)

# create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)

def mask(img,side,model_line):
    # Convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    
    # 
    se = cv2.getStructuringElement(cv2.MORPH_RECT , (4,4))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(gray, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
    out_binary = cv2.bitwise_not(out_binary)

    # Find lines
    x = np.expand_dims(gray, axis=0)
    x = x.astype('float32')
    pred = model_line.predict([x])
    pred = pred.reshape(pred.shape[1],pred.shape[2])
    pred = pred * 255
    pred = pred.astype('uint8')
    #closing
    pred = closing(pred, footprint)
    pred = cv2.bitwise_and(pred,out_binary)
    
    # find vertical line
    v_ = []
    for i in range(w-s):
        v = pred[:,i:i+s]
        v_.append(np.sum(v))
    vx = np.argmax(v_)
    if side == "left":
        if vx > p: vx = 0
        pred[:,vx:vx+1] = 255 # draw line
        pred[:,:vx] = 0 # remove left over part for extracting horizontal line
    else:
        if vx < w - p:
            vx = w
        pred[:,vx:vx+1] = 255 # draw line
        pred[:,vx:] = 0 # remove left over part for extracting horizontal line
    
    # find horizontal line
    h_ = []
    for i in range(h-s):
        h = pred[i:i+s,:]
        h_.append(np.sum(h))
    hy = np.argmax(h_)
    if hy > p:
        hy = 0
    
    pred[hy:hy+1,:] = 255
    pred[:hy,:] = 0 # remove right over part
    
    # find corner
    if side == 'left':
        top_left = (hy,vx)
        bottom_right = (h,w)
    else:
        top_left = (hy,0)
        bottom_right = (h,vx)
        
    print("vx:",vx,"hy:",hy)
    # erode
    pred = cv2.erode(pred, kernel, iterations=2)

    out = pred #pred # cv2.bitwise_and(img,img,mask = pred)
    
    # if side == "left":
    #     out = img[hy:,vx:]
    # else:
    #     out = img[hy:,:vx]

    return out

for i,image in enumerate(images):
    # read image
    print(i+1,"/",n,":",image)
    path = os.path.join(src,image)
    img = cv2.imread(path)
    # generate mask
    out = mask(img,side,model_line)
    #write out
    path = os.path.join(dst,image)
    cv2.imwrite(path,out)