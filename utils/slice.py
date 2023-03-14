import cv2
import os
import shutil
import random
import numpy as np
src = 'test/crack/right'
dst = 'sliced'
if not os.path.exists(dst):
    os.mkdir(dst)
else:
    shutil.rmtree(dst, ignore_errors=True)
    os.mkdir(dst)

def slice(img,k):
    h,w,c = img.shape
    index = []
    pieces = []
    count = 0
    for i in range(0,h,k):
        for j in range(0,w,k):
            window = img[i:i+k,j:j+k]
            index.append((i,j))
            pieces.append(window)
            #print(f'({i}:{j}) {np.sum(window)}')
            cv2.imwrite(f'{dst}/{i}_{j}.jpg',window)
            count+=1
    return pieces,index

images = os.listdir(src)
image = random.choice(images)
path = os.path.join(src,image)
img = cv2.imread(path)
pieces,index = slice(img,k=32)
fg = pieces[-1]
for i,p in enumerate(pieces):
    fg_lab = cv2.cvtColor(fg,cv2.COLOR_BGR2LAB)
    p_lab = cv2.cvtColor(p,cv2.COLOR_BGR2LAB)
    t = fg_lab - p_lab
    t = np.sum(np.power(t,2))/(p_lab.shape[0]*p_lab.shape[1])
    print(index[i],t)
    
    