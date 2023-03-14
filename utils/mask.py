import cv2
import os
import sys
import shutil
import numpy as np
from keras.models import load_model
from skimage.morphology import closing
from skimage.morphology import disk

# init
kernel = np.ones((2, 2), np.uint8)
s = 3 # window side
p = 50 # patient value
K = 124
R = 40 # 40
DIM = 256
side = "e"
#src = 'test/crack'
src = 'test/noise'
#src = 'test/good'
src = os.path.join(src,side)
dst = 'output/mask'

# load model
model_line_dir = "model/unet/lines/model.hdf5"
model_line = load_model(model_line_dir)

# read all images
images= os.listdir(src)
n = len(images)

# create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)
else:
    shutil.rmtree(dst, ignore_errors=True)
    os.mkdir(dst)

def filter(img,interation):
    s = 2
    h,w = img.shape
    for k in range(interation):
        for i in range(0,h,s):
            for j in range(0,w,s):
                window = img[i:i+s,j:j+s]
                t = np.sum(window)/255
                if t < s*s:
                    img[i:i+s,j:j+s] = 0
    return img

def outer_filter(img,interation):
    h,w = img.shape
    for k in range(interation):
        for i in range(0,h):
            for j in range(0,w):
                window = img[i-1:i+1,j-1:j+1]
                t = np.sum(window)/255
                if t == 1:
                    img[i,j] = 0
    return img
    
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
    pred = closing(pred, disk(3))
    
    # find vertical line
    v_ = []
    for i in range(w-s):
        v = pred[:,i:i+s]
        v_.append(np.sum(v))
    vx = np.argmax(v_)
    if side == "left":
        if vx > p: vx = 0
    else:
        if vx < w - p:
            vx = w
    
    # find horizontal line
    h_ = []
    for i in range(h-s):
        h = pred[i:i+s,:]
        h_.append(np.sum(h))
    hy = np.argmax(h_)
    if hy > p:
        hy = 0
    
    # remove outline
    pred[:hy,:] = 0 
    if side == "left":
        pred[:,:vx] = 0
    else:
        pred[:,vx:] = 0
    
    # grabcut
    mask = pred.copy()
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    if side == 'left':
        rect = (hy,vx,DIM-hy,DIM-vx)
    else:
        rect = (hy,0,DIM-hy,vx) 
    new_mask,fg,bg = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((new_mask==2)|(new_mask==0),0,1).astype('uint8') 
    img = img*mask2[:,:,np.newaxis]
    
    out = cv2.bitwise_or(pred,mask2*255)

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