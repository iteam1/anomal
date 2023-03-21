import sys
import cv2
import numpy as np

mask = sys.argv[1]

# Let's load a simple image with 3 black squares
image = cv2.imread('assets/'+ mask)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Find Canny edges
edged = cv2.Canny(gray, 30, 200)

# Finding Contours
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(contours)))

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
  
# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (255,0, 0), 3)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.imshow('Contours', image)
k = cv2.waitKey(0)
cv2.destroyAllWindows()