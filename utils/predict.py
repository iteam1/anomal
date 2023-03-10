'''
python3 utils/predict.py --openvino --path samples/good --model padim
'''
import os
import sys
import cv2
import time
import random
import numpy as np
from argparse import ArgumentParser,Namespace
from anomalib.post_processing import ImageResult
#from anomalib.post_processing import Visualizer
from anomalib.deploy import TorchInferencer, OpenVINOInferencer

def get_args() -> Namespace:
    '''
    Get command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('--model',type=str,default = 'padim',choices=["padim",'reverse_distillation','stfpm'], help = 'Name of the trained model')
    parser.add_argument('--openvino',action='store_true',help='Option optmize by openvino')
    parser.add_argument('--dim',type=int,default=256,help='Image crop size')
    parser.add_argument('--path',type=str,default='samples/crack',help='Path of Predict Image')
    parser.add_argument('--thresh',type=float,default=0.5,help='Anomaly threshold')
    args = parser.parse_args()
    return args

def heatmap(anomaly_map,model):
    '''
    Convert anomaly map to head map
    Args:
        anomaly_mal: anomal array score
        model: name of model
    '''
    if model == "reverse_distillation" or model == "stfpm":
        min_val = 0.3
        max_val = 0.6
    else:
        min_val = 0.0
        max_val = 1.0
    heat_map = (anomaly_map - min_val)/(max_val - min_val)
    heat_map = heat_map*255
    heat_map = heat_map - heat_map.min()
    heat_map = heat_map.astype('uint8')
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    return heat_map

def visualize(args,model,prediction):
    '''
    Handle model prediction
    Args:
        args: command line arguments
        model: model name
        prediction: prediction
    Return:
        output: post processed image (heat map or segmentation)
        pred_label: prediction label
        pred_score: prediction score
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX # font
    fontScale = 0.6 # fontScale
    color = (0,255,255)# Blue color in BGR
    thickness = 1 # Line thickness

    # Extract prediction's components
    anomaly_map = prediction.anomaly_map # anomaly map np.array (256,256) (0-1)
    heat_map = heatmap(anomaly_map,model)
    box_labels = prediction.box_labels
    gt_boxes = prediction.gt_boxes
    gt_mask = prediction.gt_mask
    image = prediction.image  # orignal image np.array (256,256,3)
    pred_boxes = prediction.pred_boxes
    pred_label = prediction.pred_label # Predict label Normal,Anomalous
    pred_mask = prediction.pred_mask # binary map np.array (256,256)
    pred_score = prediction.pred_score # predict score (0.0-1.0)
    # segmentations = prediction.segmentations

    # model customize
    if 'dfm' in model and args.openvino:
        pred_label = pred_label[0]
        if pred_label:
            pred_label = "Anomalous"
        else:
            pred_label = "Normal"
        output = image # dfm doest not have head map

    elif ('cfa' in model or 'padim' in model) and args.openvino:
        if pred_label:
            pred_label = "Anomalous"
        else:
            pred_label = "Normal"
        output = prediction.segmentations #prediction.heat_map

    elif ('reverse_distillation' in model or 'stfpm' in model):
        if pred_label:
            pred_label = "Anomalous"
        else:
            pred_label = "Normal"
        output = prediction.segmentations
    else:
        output = prediction.segmentations #prediction.heat_map

    # post process output and heatmap
    h,w,c = output.shape
    org = (5,h-20)
    text = pred_label + ":" + str(round(pred_score,4))
    output = cv2.putText(output,text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    output = cv2.addWeighted(output,0.4,heat_map,0.6,0)
    return output,pred_label,pred_score,pred_mask

# initialize
args = get_args()
model_name = args.model
model_name = ''.join(c for c in model_name if not c.isdigit())
path = args.path
DIM = args.dim
ANORMAL_THRESHOLD = args.thresh
normal = 0
anormal = 0

# visualizer
# visualizer = Visualizer(mode="simple",task="segmentation")
# directory of ouput image
image_path = f'results/{model_name}/mvtec/laptop/run/images'

if args.openvino:
    config_path = f'model/{model_name}/mvtec/laptop/run/config.yaml'
    weight_path = f'model/{model_name}/mvtec/laptop/run/openvino/model.onnx'
    meta_data_path = f'model/{model_name}/mvtec/laptop/run/openvino/meta_data.json'
    device = 'CPU' #["CPU", "GPU", "VPU"]
    inferencer = OpenVINOInferencer(config = config_path,path = weight_path,meta_data_path = meta_data_path,device = device)
else:
    config_path = f'model/{model_name}/mvtec/laptop/run/config.yaml'
    model_path = f'model/{model_name}/mvtec/laptop/run/weights/model.ckpt'
    inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

images = os.listdir(path)

if __name__ == "__main__":

    for i,image in enumerate(images):
        # Start counting time
        start_time = time.time()
        id = image.split('_')[0]
        img = cv2.imread(os.path.join(path,image))
        h,w,_ = img.shape
        top_left = img[0:DIM,0:DIM]
        top_right = img[0:DIM,w-DIM:w]

        # predict top left
        prediction = inferencer.predict(image=top_left)
        output,pred,score,mask = visualize(args,model_name,prediction)
        if pred == "Anomalous" and score > ANORMAL_THRESHOLD:
            anormal +=1
            # predict image
            name = id + '_left.jpg'
            cv2.imwrite(os.path.join(image_path,name),output)
            # mask image
            name = id + '_mask_left.jpg'
            cv2.imwrite(os.path.join(image_path,name),mask)
        else:
            normal +=1

        # predict top right
        prediction = inferencer.predict(image=top_right)
        output,pred,score,mask = visualize(args,model_name,prediction)
        if pred == "Anomalous" and score > ANORMAL_THRESHOLD:
            anormal +=1
            # predict image
            name = id + '_right.jpg'
            cv2.imwrite(os.path.join(image_path,name),output)
            # mask image
            name = id + '_mask_right.jpg'
            cv2.imwrite(os.path.join(image_path,name),mask)
        else:
            normal +=1

        end_time = time.time() - start_time
        print(i,"Inference timing consumption (s):",end_time)

    # Summary
    print("total image:",len(images))
    print("total predictions",len(images)*2)
    print("total nomal:",normal)
    print("total anomal:",anormal)
