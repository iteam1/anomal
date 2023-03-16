import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread('test/noise/right/081.png')

img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 7
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))

cv2.imwrite('result.jpg',result_image)