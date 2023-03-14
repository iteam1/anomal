import cv2
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V
from dinknet import DinkNet34

SHAPE = (256,256)
side = 'right'
#src = 'test/crack'
#src = 'test/good'
src = 'test/noise'
src = os.path.join(src,side)
dst = 'output/mask'
# create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)
else:
    shutil.rmtree(dst, ignore_errors=True)
    os.mkdir(dst)

class TTAFrame():
    def __init__(self, net):
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net().to(self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        #self.net.load_state_dict(torch.load(path))

    def predict(self, img):
        img = cv2.resize(img, SHAPE)
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
        mask3[mask3 > 4.0] = 255
        mask3[mask3 <= 4.0] = 0
        return mask3

solver = TTAFrame(DinkNet34)
solver.load('model/dsi/log01_dink34.th')

if __name__ == '__main__':
    images = os.listdir(src)
    for image in images:
        path = os.path.join(src,image)
        img = cv2.imread(path)
        mask = solver.predict(img)
        path = os.path.join(dst,image)
        cv2.imwrite(path,mask)