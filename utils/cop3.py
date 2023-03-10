'''
copy image follow mask folder
CMD: python3 utils/cop3.py datasets/laptop/train/good datasets/unet/mask0 datasets/unet/img
'''
import os
import sys
import shutil

src = sys.argv[1]
med = sys.argv[2]
dst = sys.argv[3]

# create destination folder
if not os.path.exists(dst):
    os.mkdir(dst)
    
# images
images = os.listdir(med)

for image in images:
    path_src =  os.path.join(src,image)
    path_dst = os.path.join(dst,image)
    print("copying:",path_src)
    shutil.copy(path_src,path_dst)
    