'''
Tool copy original images and save to destination folder, run from ./dates
CMD: python3 imcop.py
'''
import os
import sys
import shutil
import random

m = int(sys.argv[1])

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

n = list(range(m))

for date in dates:
    print('Searching on:',date)
    items = os.listdir(date)
    for item in items:
        path = os.path.join(date,item)
        a = os.path.isdir(path)
        b = len(item) == 14
        c = item.isalnum()
        d = random.choice(n) == 1 # random sampling
        # if all the conditions are sartified
        if a and b and c and d:
            file = os.path.join(date,item,'image_crop/top_crop.jpg')
            e = os.path.exists(file)
            if e:
                name = str(item) + "_top_crop.jpg"
                shutil.copy(file,os.path.join(dst,name))
                print('coping:',item)
                
# compress files
cmd = 'tar -zcvf '+dst+'.tar.gz'+' '+dst
print("Executing ",cmd)
os.system(cmd)
# remove destination foler
shutil.rmtree(dst)
print('Done!')
            
            
