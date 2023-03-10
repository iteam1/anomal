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

# load model
trained_model = load_model(model_dir)

# predicting images
image = random.choice(images)
path = os.path.join(src,image)
img = cv2.imread(path,0)
x = np.expand_dims(img, axis=0)
x = x.astype('float32')
print(x.shape)

pred = trained_model.predict([x])
pred = pred.reshape(pred.shape[1],pred.shape[2])
pred = pred * 255
pred = pred.astype('uint8')
print(pred.shape,pred.min(),pred.max())

out = np.hstack([pred,img])

cv2.imshow('out',out)
k = cv2.waitKey()
cv2.destroyAllWindows()