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
    parser.add_argument('--model',type=str,default = 'padim',
                        choices=["cfa", "cflow", "dfkde","dfm","padim",'patchcore','reverse_distillation','stfpm'],
                        help = 'Name of the trained model')
    parser.add_argument('--openvino',action='store_true',help='Option optmize by openvino')
    parser.add_argument('--dim',type=int,default=256,help='Image crop size')
    parser.add_argument('--path',type=str,default='samples/crack',help='Path of Predict Image')
    args = parser.parse_args()
    return args

def visualize(model,prediction):
    '''
    Handle model prediction
    Args:
        model: model name
        prediction: prediction
    Return:
        output: post processed image (heat map or segmentation)
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
    heat_map = prediction.heat_map # head map np.array (256,256,3)
    image = prediction.image  # orignal image np.array (256,256,3)
    pred_boxes = prediction.pred_boxes
    pred_label = prediction.pred_label # Predict label Normal,Anomalous
    pred_mask = prediction.pred_mask # binary map np.array (256,256)
    pred_score = prediction.pred_score # predict score (0.0-1.0)
    segmentations = prediction.segmentations
    if model == 'cfm':
        output = segmentations
    else:
        output = heat_map
    # post process heatmap
    h,w,c = heat_map.shape
    org = (5,h-20)
    text = pred_label + ":" + str(round(pred_score,2))
    output = cv2.putText(output,text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    return output

# initialize
args = get_args()
model_name = args.model
path = args.path
DIM = args.dim

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
image = random.choice(images)

if __name__ == "__main__":
    # Start counting time
    start_time = time.time()
    img = cv2.imread(os.path.join(path,image))
    h,w,_ = img.shape
    top_left = img[0:DIM,0:DIM]
    top_right = img[0:DIM,w-DIM:w]

    # predict top left
    prediction = inferencer.predict(image=top_left)
    output = visualize(model_name,prediction)
    path = os.path.join(image_path,'top_left.jpg')
    cv2.imwrite(path,output)
    
    # predict top right
    prediction = inferencer.predict(image=top_right)
    output = visualize(model_name,prediction)
    path = os.path.join(image_path,'top_right.jpg')
    cv2.imwrite(path,output)

    end_time = time.time() - start_time
    print("Inference timing consumption (s):",end_time)
