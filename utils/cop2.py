'''
Tool copy original images and save to destination folder, run from ./dates
CMD: python3 imcop.py
'''
import os
import shutil
import random

# create samples folder
dst = 'samples'
if not os.path.exists(dst):
    os.mkdir(dst)
    print('Destination folder created!')
else:
    print(dst,'Already existed, Please remove')
    exit(-1)
    
# list all dates
dates  = []
things = os.listdir()
for thing in things:
    a = os.path.isdir(thing)
    b = len(thing) == 10
    c = thing.count('_') == 2
    if a and b and c:
        dates.append(thing)

n = [1,2,3]

for date in dates:
    print('Searching on:',date)
    items = os.listdir(date)
    for item in items:
        path = os.path.join(date,item)
        a = os.path.isdir(path)
        b = len(item) == 14
        c = item.isalnum()
        d =  random.choice(n) == 1 # random sampling
        # if all the conditions are sartified
        if a and b and c and d:
            # original top
            file1 = os.path.join(date,item,'pictures/top.jpg')
            e1 = os.path.exists(file1)
            # top mask
            file2 = os.path.join(date,item,'image_crop/top/image_mask_crop_final.jpg')
            e2 = os.path.exists(file2)
            # top crop
            file3 = os.path.join(date,item,'image_crop/top_crop.jpg')
            e3 = os.path.exists(file3)
            if e1 and e2 and e3:
                print('coping:',item)
                # copy original top
                name = str(item) + "_top.jpg"
                shutil.copy(file1,os.path.join(dst,name))
                # copy mask
                name = str(item) + "_mask.jpg"
                shutil.copy(file2,os.path.join(dst,name))
                # copy top crop
                name = str(item) + "_crop.jpg"
                shutil.copy(file3,os.path.join(dst,name))
# compress files
cmd = 'tar -zcvf '+dst+'.tar.gz'+' '+dst
print("Executing ",cmd)
os.system(cmd)
# remove destination foler
shutil.rmtree(dst)
print('Done!')
            
            
