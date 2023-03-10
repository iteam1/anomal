'''
CMD: python3 utils/preprocess.py
'''
import cv2
import os
import random
import numpy as np

src = 'samples/good'
dst = 'dst'
DIM = 256
PAD = 0
L = 50
k = 3

path_prototxt = "model/hed/deploy.prototxt"
path_caffemodel = "model/hed/hed_pretrained_bsds.caffemodel"
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

# Load the model.
net = cv2.dnn.readNetFromCaffe(path_prototxt, path_caffemodel)
cv2.dnn_registerLayer('Crop', CropLayer)

# read image
images = os.listdir(src)
image =  random.choice(images)
print('Image',image)
path = os.path.join(src,image)
img = cv2.imread(path)
h,w,c = img.shape

# select region
top_left = img[0:DIM,0:DIM]
top_right = img[0:DIM,w-DIM:w] # y,x
top_left_org = top_left.copy()
top_right_org = top_right.copy()

# resize
top_left = cv2.resize(top_left,(124,124),interpolation=cv2.INTER_AREA)
top_right = cv2.resize(top_right,(125,124),interpolation=cv2.INTER_AREA)

#blur
s = 3
top_left = cv2.medianBlur(top_left,s)
top_right = cv2.medianBlur(top_right,s)

# edge detection
inp = cv2.dnn.blobFromImage(top_left, scalefactor=1.0, size=(DIM,DIM),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
net.setInput(inp)
top_left_hed = net.forward()
top_left_hed = top_left_hed[0, 0]
top_left_hed = cv2.resize(top_left_hed,(top_left_hed.shape[1],top_left_hed.shape[0]))
top_left_hed = 255 * top_left_hed
top_left_hed = top_left_hed.astype(np.uint8)

inp = cv2.dnn.blobFromImage(top_right, scalefactor=1.0, size=(DIM,DIM),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
net.setInput(inp)
top_right_hed = net.forward()
top_right_hed = top_right_hed[0, 0]
top_right_hed = cv2.resize(top_right_hed,(top_right_hed.shape[1],top_right_hed.shape[0]))
top_right_hed = 255 * top_right_hed
top_right_hed = top_right_hed.astype(np.uint8)


top_left_canny = cv2.Canny(cv2.cvtColor(top_left,cv2.COLOR_BGR2GRAY),top_left.shape[0],top_left.shape[1])
top_right_canny = cv2.Canny(cv2.cvtColor(top_right,cv2.COLOR_BGR2GRAY),top_right.shape[0],top_right.shape[1])
# top_left_canny = cv2.Canny(top_left,127,255)
# top_right_canny = cv2.Canny(top_right,127,255)

# Creating kernel
kernel = np.ones((3,3), np.uint8)
top_left_hed = cv2.erode(top_left_hed, kernel) 
top_left_hed = cv2.erode(top_left_hed, kernel)
# top_left_canny = cv2.dilate(top_left_canny, kernel) 
# top_right_canny = cv2.dilate(top_right_canny, kernel) 

top_left =  top_left_canny #cv2.bitwise_or(top_left_hed,top_left_canny)
top_right = top_right_canny #cv2.bitwise_or(top_right_hed,top_right_canny)

# downsize
top_left = cv2.resize(top_left,(DIM,DIM),interpolation=cv2.INTER_AREA)
top_right = cv2.resize(top_right,(DIM,DIM),interpolation=cv2.INTER_AREA)

top_left = top_left.astype(np.uint8)
top_right = top_right.astype(np.uint8)

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
v_l_max = np.argmax(v_left_sum) + PAD # x
h_l_max = np.argmax(h_left_sum)  + PAD # y
v_r_max = np.argmax(v_right_sum) + PAD # x
h_r_max = np.argmax(h_right_sum) + PAD # y
v_l_p = (0,v_l_max)
h_l_p = (h_l_max,0)
v_r_p = (0,v_r_max)
h_r_p = (h_r_max,0)

# check non-exist line
if v_l_max > L:
    v_l_max = 0
if v_r_max < DIM - L:
    v_r_max = 0
if h_l_max > L:
    h_l_max = 0
if h_r_max > L:
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

# shift out
steps = 50
for step in range(steps):
    M = np.float32([[1, 0, step],
                    [0, 1, 0]
                    ])
    shift_right = cv2.warpAffine(top_right, M, (top_right.shape[1], top_right.shape[0]))
    top_right = cv2.bitwise_or(top_right,shift_right)
    
for step in range(steps):
    M = np.float32([[1, 0, -step],
                    [0, 1, 0]
                    ])
    shift_left = cv2.warpAffine(top_left, M, (top_left.shape[1], top_left.shape[0]))
    top_left = cv2.bitwise_or(top_left,shift_left)

top_left = cv2.bitwise_not(top_left)
top_right = cv2.bitwise_not(top_right)

# convert to bgr
#top_left = cv2.cvtColor(top_left,cv2.COLOR_GRAY2BGR)
#top_right = cv2.cvtColor(top_right,cv2.COLOR_GRAY2BGR)

#left_out = cv2.addWeighted(top_left_org,0.3,top_left,0.7,0)
#right_out = cv2.addWeighted(top_right_org,0.3,top_right,0.7,0)    
left_out = cv2.bitwise_and(top_left_org,top_left_org,mask = top_left)
right_out = cv2.bitwise_and(top_right_org,top_right_org,mask = top_right)

mask = cv2.hconcat([left_out,right_out])

print('left vertical line:',v_l_p)
print('left horizontal line:',h_l_p)
print('right vertical line:',v_r_p)
print('right horizontal line:',h_r_p)

cv2.imwrite('assets/mask.jpg',mask)
# cv2.imwrite('assets/top_left.jpg',top_left)
# cv2.imwrite('assets/top_right.jpg',top_right)
# cv2.imwrite('assets/left_out.jpg',left_out)
# cv2.imwrite('assets/right_out.jpg',right_out)