'''
Convert format image
CMD: python3 utils/2png.py datasets/hazelnut_toy/train/good
'''
import cv2
import os
import sys
import shutil


src = sys.argv[1]

imgs = os.listdir(src)

for img in imgs:
    path = os.path.join(src,img)
    image = cv2.imread(path)
    name = img.split('.')[0]
    path2 = os.path.join(src,name+'.png')
    cv2.imwrite(path2,image)
    os.remove(path)