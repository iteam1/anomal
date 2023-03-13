import os
import cv2
import shutil

src = 'samples'
path = os.path.join(src,'A')
if not os.path.exists(path):
    os.mkdir(path)
    print(path,' created!')
    
path = os.path.join(src,'B')
if not os.path.exists(path):
    os.mkdir(path)
    print(path,' created!')
    
path = os.path.join(src,'ooo')
if not os.path.exists(path):
    os.mkdir(path)
    print(path,' created!')

# list all samples
samples = os.listdir(src)

for sample in samples:
    if ".jpg" in sample or ".png" in sample:
        path = os.path.join(src,sample)
        # read image
        img = cv2.imread(path)
        resized = cv2.resize(img,(0,0),fx=0.3,fy=0.3)
        # display
        cv2.imshow('sample',resized)
        k = cv2.waitKey()
        if k == ord('g'):
            shutil.move(path,os.path.join(src,'A'))
            cv2.destroyAllWindows()
        elif k == ord('b'):
            shutil.move(path,os.path.join(src,'B'))
            cv2.destroyAllWindows()
        elif k == ord('o'):
            shutil.move(path,os.path.join(src,'ooo'))
            cv2.destroyAllWindows()
        elif k == ord('x'):
            cv2.destroyAllWindows()
            print('Exit!')
            break
        else:
            cv2.destroyAllWindows()
    else:
        continue