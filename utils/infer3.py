import os
import sys
import cv2
import pickle
import numpy as np
from keras.models import load_model
from anomalib.deploy import TorchInferencer

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

def mask(img,model_line,model_border):
    # Convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(gray, axis=0)
    x = x.astype('float32')
    
    # Find lines
    pred_line = model_line.predict([x])
    pred_line = pred_line.reshape(pred_line.shape[1],pred_line.shape[2])
    pred_line = pred_line * 255
    pred_line = pred_line.astype('uint8')
    
    # Find border
    pred_border = model_border.predict([x])
    pred_border = pred_border.reshape(pred_border.shape[1],pred_border.shape[2])
    pred_border = pred_border * 255
    pred_border = pred_border.astype('uint8')
    
    # Combine
    pred = pred_line + pred_border
    pred = cv2.bitwise_not(pred)
    
    out =  cv2.bitwise_and(img,img,mask = pred)
    return out
    
# init
#src = 'test/crack'
src = 'test/noise'
#src = 'test/good'
side =  'right'
src = os.path.join(src,side)
dst = 'results'
THRESH = 0.5
DIM = 256
s = 3
k = 124
T = 500
v = 1
count = 0 # count anomalous

# model anomal
config_path = 'model/stfpm/mvtec/laptop/run/config.yaml'
model_path = 'model/stfpm/mvtec/laptop/run/weights/model.ckpt'
inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

# model for line and border
model_border_dir = "model/unet/border/model.hdf5"
model_line_dir = "model/unet/lines/model.hdf5"
model_line = load_model(model_line_dir)
model_border = load_model(model_border_dir)

if __name__ == "__main__":
    # list all images
    images = os.listdir(src)
    n = len(images)
    # predict
    for i,image in enumerate(images):
        # read input image
        print(i+1,"/",n,":",image)
        path = os.path.join(src,image)
        img = cv2.imread(path)
        
        # mask
        out = mask(img,model_line,model_border)
        path = os.path.join(dst,image)
        
        # first predict
        prediction = inferencer.predict(image=img)
        pred_label = prediction.pred_label
        pred_score = prediction.pred_score
        
        if pred_label == 'Anomalous' and pred_score > THRESH:
            # retest
            prediction = inferencer.predict(image=out)
            pred_label = prediction.pred_label
            pred_score = prediction.pred_score
            if pred_label == 'Anomalous' and pred_score > THRESH:
                # post process
                out = post_process(prediction)
                count +=1
                # write out
                cv2.imwrite(path,out)

    print('Total anomalous:',count)