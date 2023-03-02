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
    
# list all dates
dates  = []
things = os.listdir()
for thing in things:
    a = os.path.isdir(thing)
    b = len(thing) == 10
    c = thing.count('_') == 2
    if a and b and c:
        dates.append(thing)

n = [1,2,3,4]

for date in dates:
    print(date)
    items = os.listdir(date)
    for item in items:
        a = os.path.isdir(item)
        b = len(item) == len('05052022014305')
        c = item.isalnum()
        d = random.choice(n) == 1 # random sampling
        # if all condition sartified
        if a and b and c and d:
            file = os.path.join(date,item,'pictures/top.jpg')
            if os.isfile(file):
                name = str(item) + "_top.jpg"
                shutil.copy(file,os.path.join(dst,name))
                print('coping:',file)
            file = os.path.join(date,item,'image_crop/top_crop.jpg')
            if os.isfile(file):
                name = str(item) + "_top_crop.jpg"
                shutil.copy(file,os.path.join(dst,name))
                print('coping:',file)
                
# compress files
cmd = 'tar -zcvf '+dst+'.tar.gz'+' '+dst
print("Executing ",cmd)
os.system(cmd)

print('Done!')
            
            