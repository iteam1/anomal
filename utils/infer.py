'''
env/bin/python3 utils/infer.py -w -i path/to/your/image.jpg
python3 utils/infer.py -w -i path/to/your/image.jpg
'''
import cv2
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from dinknet import DinkNet34
from torch.autograd import Variable as V
from anomalib.deploy import TorchInferencer
from skimage.morphology import closing,disk

# init parser
parser = argparse.ArgumentParser(description = 'Anomal detection')
# add argument to parser
parser.add_argument('-i','--img',type = str, help = 'directory to image', required = True)
# parser.add_argument('-d','--dest',type = str, help = 'directory to save json file', required = True)
parser.add_argument('-w','--write',action = 'store_true', help = 'option to save debug image',required=False)
# create arguments
args = parser.parse_args()

# initialize
THRESH1 = 0.50 # for inferencer anomal
THRESH2 = 0.53 # for checker anomal
THRESH3 = 0.5 # anomalt threhsold
TOTAL = 450 # total anomaly score threshold
P = 10
K = 48 # corner window size
DIM = 256 # image dimension size
SHAPE = (DIM,DIM) # shape of image
T = 250 # threshold of total white pixel range
label = 'normal' # prediction label
postions = [] # crack position
rois = [] # crack region (x,y,w,h) left side and right side
class TTAFrame():
    def __init__(self, net):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net().to(self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        #self.net.load_state_dict(torch.load(path))

    def predict(self, img):
        img = cv2.resize(img, SHAPE)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).to(self.device))
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        # post process
        threshold = 3.0
        mask3[mask3 > threshold] = 255
        mask3[mask3 <= threshold] = 0
        return mask3

def heatmap(anomaly_map):
    min_val = 0.3
    max_val = 0.6
    heat_map = (anomaly_map - min_val)/(max_val - min_val)
    heat_map = heat_map*255
    heat_map = heat_map - heat_map.min()
    heat_map = heat_map.astype('uint8')
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    return heat_map

def post_process(prediction):
    # Init
    font = cv2.FONT_HERSHEY_SIMPLEX # font
    fontScale = 0.6 # fontScale
    color = (0,255,255)# Blue color in BGR
    thickness = 1 # Line thickness

    # Extract prediction's components
    anomaly_map = prediction.anomaly_map
    heat_map = heatmap(anomaly_map)
    box_labels = prediction.box_labels
    gt_boxes = prediction.gt_boxes
    gt_mask = prediction.gt_mask
    image = prediction.image
    pred_boxes = prediction.pred_boxes
    pred_label = prediction.pred_label
    pred_mask = prediction.pred_mask
    pred_score = prediction.pred_score
    output = prediction.segmentations

    # Post process output and heatmap
    h,w,c = output.shape
    org = (5,h-20)
    text = pred_label + ":" + str(round(pred_score,4))
    output = cv2.putText(output,text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    output = cv2.addWeighted(output,0.6,heat_map,0.4,0)
    return output

def RLSA_X(img_src, zero_length):
    hor_thres = zero_length
    zero_count = 0
    one_flag = 0

    if img_src is None:
        return None

    #img_src = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

    tmpImg = np.copy(img_src)

    for i in range(img_src.shape[0]):
        one_flag = 0
        zero_count = 0
        for j in range(img_src.shape[1]):
            if img_src[i, j] == 255:
                if one_flag == 255:
                    if zero_count <= hor_thres and zero_count <= j:
                        cv2.line(tmpImg, (j - zero_count, i), (j, i), (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 255
            else:
                if one_flag == 255:
                    zero_count = zero_count + 1

    _, tmpImg = cv2.threshold(tmpImg, 100, 255, cv2.THRESH_BINARY)

    return tmpImg

def RLSA_Y(img_src, zero_length):
    hor_thres = zero_length
    zero_count = 0
    one_flag = 0

    if img_src is None:
        return None
    
    tmpImg = np.copy(img_src)

    for i in range(img_src.shape[1]):
        one_flag = 0
        zero_count = 0
        for j in range(1, img_src.shape[0]):
            if img_src[j, i] == 255:
                if one_flag == 255:
                    if zero_count <= hor_thres and zero_count <= j:
                        cv2.line(tmpImg, (i, j - zero_count), (i, j), (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 255
            else:
                if one_flag == 255:
                    zero_count = zero_count + 1

    _, tmpImg = cv2.threshold(tmpImg, 100, 255, cv2.THRESH_BINARY)

    return tmpImg
    
def gen_mask(img,mask,side):
    height,width = mask.shape
    s = 2
    p = 40
    # find vertical line
    v_ = []
    for i in range(width-s):
        v = mask[:,i:i+s]
        v_.append(np.sum(v))
    vx = np.argmax(v_)
    if side == "left":
        if vx > p: vx = 0
        mask[:,vx:vx+1] = 0 # draw line
        mask[:,:vx] = 0 # remove left over part for extracting horizontal line
    else:
        if vx < width - p:
            vx = width
        mask[:,vx:vx+1] = 0 # draw line
        mask[:,vx:] = 0 # remove left over part for extracting horizontal line
    # patient
    if side == "left":
        if vx > p:
            vx = 0
    else:
        if vx < width - p:
            vx  =width
    
    # find horizontal line
    h_ = []
    for i in range(height-s):
        h = mask[i:i+s,:]
        h_.append(np.sum(h))
    hy = np.argmax(h_)
    # patience
    if hy > p:
        hy = 0
    
    mask[hy:hy+1,:] = 0
    mask[:hy,:] = 0 # remove right over part
    
    if side == "left":
        mask = mask[hy:,vx:]
        roi = img[hy:,vx:]
    else:
        mask = mask[hy:,:vx]
        roi = img[hy:,:vx]
    
    return mask,(hy,vx),roi

def fillin(mask,cord):
    hy,vx = cord[0],cord[1]
    # fillin
    H,W = mask.shape
    for i in range(W-1,0,-1):
        for j in range(H-1,0,-1):
            if mask[j,i] == 255:
                break
            else:
                mask[j,i]= 255
    # horizontal fill
    kernel = np.ones((1,6), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # erode full mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask

def rounded(mask,side):
    p = 8
    d = 0
    corner = True
    h,w = mask.shape
    
    if side == 'right':
        mask = cv2.flip(mask,1)

    l_x = 1
    l_y = 1
    for i in range(1,w-1):
        point = mask[3:4,i:i+1]
        if point == 255:
            l_x = i
            break
        
    for i in range(1,h-1):
        point = mask[i:i+1,1:2]
        if point == 255:
            l_y = i
            break
        
    if l_x > K and l_y > K:
        l_y = 1
        l_x = 1
        corner = False
    
    if corner: mask = cv2.ellipse(mask,(l_y+p,l_x+p),(l_y-d,l_x-d),0,0,360,(255),-1)
        
    if side == 'right':
        mask = cv2.flip(mask,1)
        
    return mask

def mask_img(input,side,solver):
    m = solver.predict(input)
    m = RLSA_Y(m,20)
    m = RLSA_X(m,20)
    m = closing(m,disk(3))
    m,cord,roi = gen_mask(input,m,side)
    m = fillin(m,cord)
    #rounded
    m = rounded(m,side)
    m = m.astype('uint8')
    out = cv2.bitwise_and(roi,roi,mask = m)
    out = cv2.resize(out,(DIM,DIM),interpolation=cv2.INTER_AREA)
    roi = cv2.resize(roi,(DIM,DIM),interpolation=cv2.INTER_AREA)
    m = cv2.resize(m,(DIM,DIM),interpolation=cv2.INTER_AREA)
    return out,roi,m,cord

def predict(input,side,solver):
    label = None # label of prediction
    rect = None # rectangle of crack
    rects = []
    areas = []
    distances = []
    
    # resize image
    image = cv2.resize(input,(DIM,DIM),interpolation=cv2.INTER_AREA)
    out,out_color,out_mask,cord = mask_img(image,side,solver)
    
    # choose padding and top_point
    if side == 'left':
        top_point = (0,0)
        py,px = cord
    else:
        top_point = (DIM,0) # x,y
        py,px = cord
        px = - (DIM - px)
    
    # predict
    prediction = tester.predict(image=out)
    pred_label = prediction.pred_label
    pred_score = prediction.pred_score
    pred_mask = prediction.pred_mask
    final_mask = cv2.bitwise_and(out_mask,pred_mask)

    # Finding Contours
    blank = np.zeros(final_mask.shape)
    contours, hierarchy = cv2.findContours(final_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(blank, (x, y), (x + w, y + h), (255),-1)
    
    contours, hierarchy = cv2.findContours(blank.astype('uint8'),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        xc = x+w/2
        yc = y+h/2
        s = w*h
        d = np.sqrt((xc-top_point[0])**2 + (yc-top_point[1])**2)
        areas.append(s)
        distances.append(d)
        rects.append((int(x),int(y),int(w),int(h))) # convert int64 to int16 for json serialize
        # image = cv2.circle(image,(x,y),3,(255),-1)
        # image = cv2.line(image,(int(xc),int(yc)),top_point,(255),2)
        
    #cv2.imwrite('image.jpg',image)
    if len(distances) != 0:
        idx = np.argmin(distances)
        rect = rects[idx]

    # check anomaly map
    anomaly_map = prediction.anomaly_map
    anomaly_map = anomaly_map * (final_mask == 255)
    anomaly_map = anomaly_map * (anomaly_map>THRESH3)
    if pred_label == 'Anomalous' and pred_score > THRESH2:
        # check corner condition
        if side == 'left':
            anomal_value = anomaly_map[:K,:K]
            corner = final_mask[:K,:K]
        else:
            anomal_value = anomaly_map[:K,DIM-K:DIM]
            corner = final_mask[0:K,DIM-K:DIM]
        area = np.sum(corner)/255
        anomal_value = np.sum(anomal_value)
        if area > T and anomal_value > TOTAL:                
            # conclude
            label = "crack"
            return label,prediction,rect
        else:
            # conclude
            label = "normal"
            return label,prediction,rect
    else:
        # conclude
        label = "normal"
        return label,prediction,rect

# load retest anomal model
config_path = 'model/lim/config.yaml'
model_path = 'model/lim/model.ckpt'
tester = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

# load dsi model
solver = TTAFrame(DinkNet34)
solver.load('model/dsi/log01_dink34.th')

if __name__ == "__main__":
    
    # start counting time
    start_time = time.time()
        
    # read input image
    img_org = cv2.imread(args.img)
    H,W,_ = img_org.shape
    top_left = img_org[0:DIM,0:DIM]
    top_right = img_org[0:DIM,W-DIM:W]
    
    # predict top left
    label_left,prediction,rect = predict(top_left,"left",solver)
    if label_left == "crack":
        postions.append('left')
        result = post_process(prediction)
        if rect:
            x,y,w,h = rect
            cv2.rectangle(top_left, (x, y), (x + w, y + h), (0,0,255),1)
            rois.append(rect)
        if args.write:
            cv2.imwrite('top_left.jpg',top_left)
            #cv2.imwrite('corner_left.jpg',result)
    
    # predict top right
    label_right,prediction,rect = predict(top_right,"right",solver)
    if label_right == "crack":
        postions.append('right')
        result = post_process(prediction)
        if rect:
            x,y,w,h = rect
            cv2.rectangle(top_right, (x, y), (x + w, y + h), (0,0,255),1)
            x = W - DIM + x # convert to full image cordinate
            rect = (x,y,w,h)
            rois.append(rect)
        if args.write:
            cv2.imwrite('top_right.jpg',top_right)
            #cv2.imwrite('corner_right.jpg',result)
    
    # conclude
    if label_left == 'crack' or label_right == 'crack':
        label = "crack"
    
    print(args.img,' ==> ',label)
        
    # stop counting time
    end_time = time.time() - start_time
    end_time = round(end_time,3)
    
    # create result dictionary
    result = {'image':args.img,
              'prediction':label,
              'positions': postions,
              'rois':rois,
              'consuming time': end_time
              }
    
    # serializing to json
    result_json = json.dumps(result)
    
    # export json
    with open('result.json','w') as f:
        f.write(result_json)
    
    print('consuming time (s):',end_time)