'''
create mask
CMD: python3 utils/mask.py assets/mask
'''
import os
import sys
import cv2
import numpy as np

src = sys.argv[1]
dst = 'ground_truth'
lower = np.array([0,0,253])
upper = np.array([1,1,255])

if not os.path.exists(dst):
    os.mkdir(dst)

imgs = os.listdir(src)

for i in imgs:
    path = os.path.join(src,i)
    img = cv2.imread(path)
    mask = cv2.inRange(img,lower,upper)
    # cv2.imshow('mask',mask)
    # k = cv2.waitKey()
    # cv2.destroyAllWindows()
    path = os.path.join(dst,i)
    cv2.imwrite(path,mask)