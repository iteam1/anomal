'''
Crop all input image to create dataset
CMD: python3 utils/crop2.py datasets/laptop/train/good train
'''
import os
import sys
import cv2
import random

src = sys.argv[1]
dst = sys.argv[2]

DIM = 256 # 512
count = 0

if not os.path.exists(src):
    print('Samples Not Found!')
    exit(-1)
    
if not os.path.exists(dst):
    os.mkdir(dst)
    
images  = os.listdir(src)
print(len(images))

for image in images:
    print(image)
    num = image.split('.')
    num = num[0]
    num = int(num)
    path = os.path.join(src,image)
    img = cv2.imread(path)
    h,w,_ = img.shape
    if num % 2 == 0:
        top_left = img[0:DIM,0:DIM]
        path = os.path.join(dst,image)
        cv2.imwrite(path,top_left)     
    else:
        top_right = img[0:DIM,w-DIM:w]
        path = os.path.join(dst,image)
        cv2.imwrite(path,top_right)
    