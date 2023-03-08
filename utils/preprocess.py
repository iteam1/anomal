import cv2
import os
import random
import numpy as np

src = 'samples/good'
dst = 'dst'
DIM = 256
PAD = 10
images = os.listdir(src)
image = random.choice(images)
path = os.path.join(src,image)

#
img = cv2.imread(path)
h,w,c = img.shape
top_left = img[0:DIM,0:DIM]
top_right = img[0:DIM,w-DIM:w] # y,x
        
# left side
h_left = img[DIM-PAD:DIM,0:DIM]
v_left = img[0:DIM,DIM-PAD:DIM]
# right side
v_right = img[0:DIM,w-DIM:w-DIM+PAD]
h_right = img[DIM-PAD:DIM,w-DIM:w] 
# check shape
print('v_left',v_left.shape)
print('h_left',h_left.shape)
print('h_right',h_right.shape)
print('v_right',v_right.shape)

top_left = cv2.Canny(top_left,127,255)
top_right = cv2.Canny(top_right,127,255)

# cv2.imwrite('assets/v_left.jpg',v_left)
# cv2.imwrite('assets/h_left.jpg',h_left)
cv2.imwrite('assets/top_left.jpg',top_left)
# cv2.imwrite('assets/v_right.jpg',v_right)
# cv2.imwrite('assets/h_right.jpg',h_right)
cv2.imwrite('assets/top_right.jpg',top_right)
#cv2.imwrite('assets/img.jpg',img)