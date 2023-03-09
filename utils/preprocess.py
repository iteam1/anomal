import cv2
import os
import random
import numpy as np

src = 'sample1/good'
dst = 'dst'
DIM = 256
PAD = 10
k = 3

images = os.listdir(src)
image = '01112022182433_top_crop.jpg' #random.choice(images)
path = os.path.join(src,image)

#
img = cv2.imread(path)
h,w,c = img.shape
top_left = img[0:DIM,0:DIM]
top_right = img[0:DIM,w-DIM:w] # y,x


imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, 5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)


# canny edges
top_left = cv2.Canny(top_left,127,255)
top_right = cv2.Canny(top_right,127,255)
# print('top_left',top_left.shape,top_left.max(),top_left.min())
# print('top_right',top_right.shape,top_right.max(),top_right.min())

# find vertical lines
v_left_sum = []
v_right_sum = []
for i in range(DIM-k):
    v_l = top_left[:,i:i+k] # y,x
    v_r =  top_right[:,i:i+k]
    v_left_sum.append(np.sum(v_l)/255)
    v_right_sum.append(np.sum(v_r)/255)

# find horizontal lines
h_left_sum = []
h_right_sum = []
for i in range(DIM-k):
    h_l = top_left[i:i+k,:]
    h_r = top_right[i:i+k,:]
    h_left_sum.append(np.sum(h_l)/255)
    h_right_sum.append(np.sum(h_r)/255)

# find positions    
v_l_max = np.argmax(v_left_sum) # x
h_l_max = np.argmax(h_left_sum) # y
v_r_max = np.argmax(v_right_sum) # x
h_r_max = np.argmax(h_right_sum) # y
v_l_p = (0,v_l_max)
h_l_p = (h_l_max,0)
v_r_p = (0,v_r_max)
h_r_p = (h_r_max,0)

# check non-exist line
if v_l_max > 100:
    v_l_max = 0
if v_r_max < 155:
    v_r_max = 0
if h_l_max > 100:
    h_l_max = 0
if h_r_max > 100:
    h_r_max = 0
    
# draw vertical lines
# top_left[:,v_l_max:v_l_max+1] = 255
# top_right[:,v_r_max:v_r_max+1] = 255
top_left[:,:v_l_max] = 255
top_right[:,v_r_max:] = 255
# draw horizontal lines
# top_left[h_l_max:h_l_max+1,:] = 255
# top_right[h_r_max:h_r_max+1,:] = 255
top_left[:h_l_max,:] = 255
top_right[:h_r_max,:] = 255

# corner
left_mask = 255 - top_left
right_mask = 255 - top_right

print('left vertical line:',v_l_p)
print('left horizontal line:',h_l_p)
print('right vertical line:',v_r_p)
print('right horizontal line:',h_r_p)

cv2.imwrite('assets/top_left.jpg',top_left)
cv2.imwrite('assets/top_right.jpg',top_right)
cv2.imwrite('assets/left_mask.jpg',left_mask)
cv2.imwrite('assets/right_mask.jpg',right_mask)