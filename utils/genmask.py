from keras.models import load_model
import numpy as np
import cv2
import os
import random
import sys

DIM = 256
src = 'datasets/segment/train/border/image'
#src = 'datasets/segment/test'
model_border_dir = "model/unet/border/model.hdf5"
model_line_dir = "model/unet/lines/model.hdf5"

images= os.listdir(src)
print(len(images))

# load model
model_line = load_model(model_line_dir)
model_border = load_model(model_border_dir)

# predicting images
#image = random.choice(images)
for i,image in enumerate(images):
    path = os.path.join(src,image)
    img = cv2.imread(path)
    #img = cv2.medianBlur(img,7)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(gray, axis=0)
    x = x.astype('float32')
    print(x.shape)

    pred_line = model_line.predict([x])
    pred_line = pred_line.reshape(pred_line.shape[1],pred_line.shape[2])
    pred_line = pred_line * 255
    pred_line = pred_line.astype('uint8')
    #pred_line = cv2.bitwise_not(pred_line)
    print(pred_line.shape,pred_line.min(),pred_line.max())

    pred_border = model_border.predict([x])
    pred_border = pred_border.reshape(pred_border.shape[1],pred_border.shape[2])
    pred_border = pred_border * 255
    pred_border = pred_border.astype('uint8')
    print(pred_border.shape,pred_border.min(),pred_border.max())

    pred = pred_line + pred_border
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(pred,0.6,img,0.4,0)
    # cv2.imshow('out',out)
    # k = cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite(f'results/mask{i}.jpg',out)
