'''
CMD: python3 utils/view2.py
'''
import cv2
import os
import random

src = 'samples'
images  =os.listdir(src)
image = random.choice(images)
id = image.split('_')[0]
print('ID:',id)
path_img = os.path.join(src,id+"_top.jpg")
path_mask = os.path.join(src,id + "_mask.jpg")
img = cv2.imread(path_img)
mask = cv2.imread(path_mask)
#mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

output = cv2.addWeighted(img,0.5,mask,0.5,0)
cv2.imwrite('assets/out.jpg',output)