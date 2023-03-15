import cv2
import os
import torch
import numpy as np
import torch.nn as nn
from dinknet import DinkNet34
from torch.autograd import Variable as V
from anomalib.deploy import TorchInferencer
from skimage.morphology import closing,disk

# init
side = 'left'
src = 'test/noise'
src = os.path.join(src,side)
dst = 'results'
THRESH1 = 0.50 # for inferencer anomal
THRESH2 = 0.52 # for checker anomal
DIM = 256 # image dimension size
SHAPE = (DIM,DIM) # shape of image
T = 500 # threshold of total white pixel range
count = 0 # count anomalous

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
class TTAFrame():
    def __init__(self, net):
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net().to(self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
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
    
def gen_mask(mask,side):
    H,W = mask.shape
    s = 2
    p = 50
    # find vertical line
    v_ = []
    for i in range(W-s):
        v = mask[:,i:i+s]
        v_.append(np.sum(v))
    vx = np.argmax(v_)
    if side == "left":
        if vx > p: vx = 0
        mask[:,vx:vx+1] = 0 # draw line
        mask[:,:vx] = 0 # remove left over part for extracting horizontal line
    else:
        if vx < W - p:
            vx = W
        mask[:,vx:vx+1] = 0 # draw line
        mask[:,vx:] = 0 # remove left over part for extracting horizontal line
    
    # find horizontal line
    h_ = []
    for i in range(H-s):
        h = mask[i:i+s,:]
        h_.append(np.sum(h))
    hy = np.argmax(h_)
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
    kernel = np.ones((1, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def mask(img,side,solver):
    mask = solver.predict(img)
    mask = RLSA_Y(mask,20)
    mask = RLSA_X(mask,20)
    mask = closing(mask,disk(3))
    mask,cord,roi = gen_mask(mask,side)
    mask = fillin(mask,cord)
    out = cv2.bitwise_and(roi,roi,mask = mask.astype('uint8'))
    out = cv2.resize(out,(DIM,DIM),interpolation=cv2.INTER_AREA)
    return out

# load dsi model
solver = TTAFrame(DinkNet34)
solver.load('model/dsi/log01_dink34.th')

# model anomal model
config_path = 'model/stfpm/mvtec/laptop/run/config.yaml'
model_path = 'model/stfpm/mvtec/laptop/run/weights/model.ckpt'
# config_path = 'model/ooo/mvtec/ooo/run/config.yaml'
# model_path = 'model/ooo/mvtec/ooo/run/weights/model.ckpt'
inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

# load retest anomal model
config_path = 'model/ooo/mvtec/ooo/run/config.yaml'
model_path = 'model/ooo/mvtec/ooo/run/weights/model.ckpt'
tester = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

# load dsi model
solver = TTAFrame(DinkNet34)
solver.load('model/dsi/log01_dink34.th')

if __name__ == "__main__":
    # list all images
    images = os.listdir(src)
    n = len(images)
    # predict
    for i,image in enumerate(images):
        print(i+1,"/",n,":",image)
        path = os.path.join(src,image)
        # read input image
        img = cv2.imread(path)
        img = cv2.resize(img,(DIM,DIM),interpolation=cv2.INTER_AREA)

        # first predict
        prediction = inferencer.predict(image=img)
        pred_label = prediction.pred_label
        pred_score = prediction.pred_score
        
        # retest
        if pred_label == 'Anomalous' and pred_score > THRESH1:
            print('retest',image)
            # mask imput image
            out = mask(img,side,solver)
            prediction = tester.predict(image=out)
            pred_label = prediction.pred_label
            pred_score = prediction.pred_score
            if pred_label == 'Anomalous' and pred_score > THRESH2:
                out = post_process(prediction)
                path = os.path.join(dst,image)
                cv2.imwrite(path,out)
                count +=1

    print('Total anomalous:',count)
