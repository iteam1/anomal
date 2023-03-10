from keras.models import load_model
import numpy as np
import cv2
import os
import random
import sys

DIM = 256
src = 'datasets/unet/train/image'
model_dir = "model/unet/model.hdf5"
images= os.listdir(src)
print(len(images))

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

# load model
trained_model = load_model(model_dir)

# predicting images
image = random.choice(images)
path = os.path.join(src,image)
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x = np.expand_dims(gray, axis=0)
x = x.astype('float32')
print(x.shape)

pred = trained_model.predict([x])
pred = pred.reshape(pred.shape[1],pred.shape[2])
pred = pred * 255
pred = pred.astype('uint8')
print(pred.shape,pred.min(),pred.max())

# Load the model.
net = cv2.dnn.readNetFromCaffe("model/hed/deploy.prototxt","model/hed/hed_pretrained_bsds.caffemodel")
cv2.dnn_registerLayer('Crop', CropLayer)


inp = cv2.dnn.blobFromImage(img, scalefactor=2.0, size=(DIM,DIM),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
net.setInput(inp)

out = net.forward()
out = out[0, 0]
out  = out > 127
out = out * 255
out = out.astype('uint8')

pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
out = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)

print(out.shape,pred.shape,img.shape)
out = cv2.resize(out, (img.shape[1], img.shape[0]))

mask = out + pred
out = np.hstack([mask,out,pred,img])

cv2.imshow('out',out)
k = cv2.waitKey()
cv2.destroyAllWindows()