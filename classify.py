import os
import random
import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import imgaug as ia
import imgaug.augmenters as iaa
ia.seed(1)

path ='datasets/classify'
labels = os.listdir(os.path.join(path))

dic = {}
for k,l in enumerate(labels):
    dic[k]=l
print(dic)
dic_invert = {v:k for k,v in dic.items()}
print(dic_invert)

augmentation = iaa.Sequential([
    # 1. Flip
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    # 2. Linearcontrast
    iaa.LinearContrast((0.8, 1.2)),
    # 3. Perform methods below only sometimes
    iaa.Sometimes(0.5,
    # 4. GaussianBlur
    iaa.GaussianBlur((0.0, 3.0))
    )
    ])

N = 2000
X = []
y = [] #0,1,2,3,4
for k,label in enumerate(labels):
    images = os.listdir(os.path.join(path,label))
    m = len(images)
    n = N//m
    for image in images:
        img = cv2.imread(os.path.join(path,label,image),0)
        imgs = np.array([ img for _ in range(n)],dtype=np.uint8)
        imaugs = augmentation(images=imgs)
        for aug in imaugs:
            X.append(aug)
            y.append(k)

X = np.array(X).reshape(len(X),-1)
# shuffle
X,y = shuffle(X,y)
# norm
X = X/255.0
y = np.array(y)

# split
X_train, X_val, y_train, y_val = train_test_split(X,y)
print("X_train",X_train.shape)
print("X_val",X_val.shape)
print("y_train",y_train.shape)
print("y_val",y_val.shape)

svc = SVC(kernel='linear',gamma='auto') #linear,rbf
svc.fit(X_train, y_train)

pred = svc.predict(X_val)

print("Accuracy on unknown data is",classification_report(y_val,pred))

# save the model to disk
filename = 'model/classify/model.sav'
pickle.dump(svc, open(filename, 'wb'))
