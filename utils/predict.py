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
from anomalib.deploy import TorchInferencer, OpenVINOInferencer

def get_args() -> Namespace:
    '''
    Get command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('--model',type=str,default = 'padim',choices=["padim",'reverse_distillation','stfpm',
                                                                      "padim2",'reverse_distillation2','stfpm2',
                                                                      "padim3",'reverse_distillation3','stfpm3'], help = 'Name of the trained model')
    parser.add_argument('--openvino',action='store_true',help='Option optmize by openvino')
    parser.add_argument('--dim',type=int,default=256,help='Image crop size')
    parser.add_argument('--path',type=str,default='samples/crack',help='Path of Predict Image')
    args = parser.parse_args()
    return args

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
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    
    # Extract prediction's components
    anomaly_map = prediction.anomaly_map # anomaly map np.array (256,256)
    box_labels = prediction.box_labels
    gt_boxes = prediction.gt_boxes
    gt_mask = prediction.gt_mask
    image = prediction.image  # orignal image np.array (256,256,3)
    pred_boxes = prediction.pred_boxes
    pred_label = prediction.pred_label # Predict label Normal,Anomalous
    pred_mask = prediction.pred_mask # binary map np.array (256,256)
    pred_score = prediction.pred_score # predict score (0.0-1.0)
    # segmentations = prediction.segmentations
    # cv2.imwrite('mask.jpg',pred_mask) # dfm mask = None
    
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
    # post process heatmap
    h,w,c = output.shape
    org = (5,h-20)
    text = pred_label + ":" + str(round(pred_score,2))
    output = cv2.putText(output,text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    return output,pred_label,pred_score

# initialize
args = get_args()
model_name = args.model
path = args.path
DIM = args.dim
ANORMAL_THRESHOLD = 0.65
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
        img = cv2.imread(os.path.join(path,image))
        h,w,_ = img.shape
        top_left = img[0:DIM,0:DIM]
        top_right = img[0:DIM,w-DIM:w]

        # predict top left
        prediction = inferencer.predict(image=top_left)
        output,pred,score = visualize(args,model_name,prediction)
        if pred == "Anomalous" and score > ANORMAL_THRESHOLD:
            anormal +=1
            name = 'top_left_'+image
            cv2.imwrite(os.path.join(image_path,name),output)
        else:
            normal +=1
            
        # predict top right
        prediction = inferencer.predict(image=top_right)
        output,pred,score = visualize(args,model_name,prediction)
        if pred == "Anomalous" and score > ANORMAL_THRESHOLD:
            anormal +=1
            name = 'top_right_'+image
            cv2.imwrite(os.path.join(image_path,name),output)
        else:
            normal +=1

        end_time = time.time() - start_time
        print(i,"Inference timing consumption (s):",end_time)
    
    # Summary
    print("total image:",len(images))
    print("total predictions",len(images)*2)
    print("total nomal:",normal)
    print("total anomal:",anormal)
