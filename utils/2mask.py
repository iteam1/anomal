'''
create mask
CMD: python3 utils/mask.py datasets/segment/train/1/mask label
'''
import os
import sys
import cv2
import numpy as np

src = sys.argv[1]
dst = sys.argv[2]

lower = np.array([0,0,253])
upper = np.array([0,0,255])

if not os.path.exists(dst):
    os.mkdir(dst)

imgs = os.listdir(src)

for i in imgs:
    path = os.path.join(src,i)
    img = cv2.imread(path)
    mask = cv2.inRange(img,lower,upper)
    #mask = cv2.bitwise_not(mask)
    path = os.path.join(dst,i)
    cv2.imwrite(path,mask)
