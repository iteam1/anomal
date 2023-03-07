'''
Crop input image to create dataset
CMD: python3 utils/crop.py samples/good good
CMD: python3 utils/crop.py samples/crack crack
'''
import os
import sys
import cv2
import random

src = sys.argv[1]
tmp = sys.argv[2]

dst = 'datasets'
cat = 'laptop'

DIM = 256 #512
n = [1,2,3]
count = 0

def rename(count):
    name = str(count)
    if len(name)==1:
        name = "00"+name
    elif len(name)==2:
        name = "0"+name
    else:
        name = name
    name +=".png"
    return name

if not os.path.exists(src):
    print('Samples Not Found!')
    exit(-1)
    
if not os.path.exists(dst):
    os.mkdir(dst)

path = os.path.join(dst,cat)
if not os.path.exists(path):
    os.mkdir(path)
    
path = os.path.join(dst,cat,tmp)
if not os.path.exists(path):
    os.mkdir(path)
    
imgs  = os.listdir(src)

for i in imgs:
    v = random.choice(n)
    if v == 1 or v == 2:
        path = os.path.join(src,i)
        img = cv2.imread(path)
        h,w,_ = img.shape
        top_left = img[0:DIM,0:DIM]
        name = rename(count)
        path = os.path.join(dst,cat,tmp,name)
        cv2.imwrite(path,top_left)     
        count +=1
        top_right = img[0:DIM,w-DIM:w]
        name = rename(count)
        path = os.path.join(dst,cat,tmp,name)
        cv2.imwrite(path,top_right)
        count +=1

print("Total:",count)
    