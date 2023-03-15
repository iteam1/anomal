import cv2 as cv
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V
from dinknet import DinkNet34
from skimage.morphology import closing,disk

SHAPE = (256,256)
DIM = 256
side = 'right'
#src = 'test/crack'
src = 'test/good'
#src = 'test/noise'
src = os.path.join(src,side)
dst = 'output/mask'
# create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)
else:
    shutil.rmtree(dst, ignore_errors=True)
    os.mkdir(dst)

def RLSA_X(img_src, zero_length):
    hor_thres = zero_length
    zero_count = 0
    one_flag = 0

    if img_src is None:
        return None

    #img_src = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

    tmpImg = np.copy(img_src)

    for i in range(img_src.shape[0]):
        one_flag = 0
        zero_count = 0
        for j in range(img_src.shape[1]):
            if img_src[i, j] == 255:
                if one_flag == 255:
                    if zero_count <= hor_thres and zero_count <= j:
                        cv.line(tmpImg, (j - zero_count, i), (j, i), (255, 255, 255), 1, cv.LINE_AA)
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 255
            else:
                if one_flag == 255:
                    zero_count = zero_count + 1

    _, tmpImg = cv.threshold(tmpImg, 100, 255, cv.THRESH_BINARY)

    return tmpImg

def RLSA_Y(img_src, zero_length):
    hor_thres = zero_length
    zero_count = 0
    one_flag = 0

    if img_src is None:
        return None
    
    tmpImg = np.copy(img_src)

    for i in range(img_src.shape[1]):
        one_flag = 0
        zero_count = 0
        for j in range(1, img_src.shape[0]):
            if img_src[j, i] == 255:
                if one_flag == 255:
                    if zero_count <= hor_thres and zero_count <= j:
                        cv.line(tmpImg, (i, j - zero_count), (i, j), (255, 255, 255), 1, cv.LINE_AA)
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 255
            else:
                if one_flag == 255:
                    zero_count = zero_count + 1

    _, tmpImg = cv.threshold(tmpImg, 100, 255, cv.THRESH_BINARY)

    return tmpImg

class TTAFrame():
    def __init__(self, net):
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net().to(self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        #self.net.load_state_dict(torch.load(path))

    def predict(self, img):
        img = cv.resize(img, SHAPE)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(self.device))

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        
        # post process
        threshold = 3.0
        mask3[mask3 > threshold] = 255
        mask3[mask3 <= threshold] = 0
        return mask3
    
def post_process(mask,side):
    H,W = mask.shape
    s = 2
    p = 50
    # find vertical line
    v_ = []
    for i in range(W-s):
        v = mask[:,i:i+s]
        v_.append(np.sum(v))
    vx = np.argmax(v_)
    if side == "left":
        if vx > p: vx = 0
        mask[:,vx:vx+1] = 0 # draw line
        mask[:,:vx] = 0 # remove left over part for extracting horizontal line
    else:
        if vx < W - p:
            vx = W
        mask[:,vx:vx+1] = 0 # draw line
        mask[:,vx:] = 0 # remove left over part for extracting horizontal line
    
    # find horizontal line
    h_ = []
    for i in range(H-s):
        h = mask[i:i+s,:]
        h_.append(np.sum(h))
    hy = np.argmax(h_)
    if hy > p:
        hy = 0
    
    mask[hy:hy+1,:] = 0
    mask[:hy,:] = 0 # remove right over part
    
    if side == "left":
        mask = mask[hy:,vx:]
        roi = img[hy:,vx:]
    else:
        mask = mask[hy:,:vx]
        roi = img[hy:,:vx]
    
    return mask,(hy,vx),roi

def fillin(mask,cord):
    hy,vx = cord[0],cord[1]
    # fillin
    H,W = mask.shape
    for i in range(W-1,0,-1):
        for j in range(H-1,0,-1):
            if mask[j,i] == 255:
                break
            else:
                mask[j,i]= 255
    kernel = np.ones((1, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=2)
    return mask

def mask(img,side,solver):
    mask = solver.predict(img)
    mask = RLSA_Y(mask,20)
    mask = RLSA_X(mask,20)
    mask = closing(mask,disk(3))
    mask,cord,roi = post_process(mask,side)
    mask = fillin(mask,cord)
    out = cv.bitwise_and(roi,roi,mask = mask.astype('uint8'))
    out = cv.resize(out,(DIM,DIM),interpolation=cv.INTER_AREA)
    return out

# load dsi model
solver = TTAFrame(DinkNet34)
solver.load('model/dsi/log01_dink34.th')

if __name__ == '__main__':
    images = os.listdir(src)
    for i,image in enumerate(images):
        path = os.path.join(src,image)
        print(f'{i}/{len(images)}:{image}')
        # read image
        img = cv.imread(path)
        # mask
        out = mask(img,side,solver)
        # write out
        path = os.path.join(dst,image)
        cv.imwrite(path,out)