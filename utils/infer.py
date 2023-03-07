import os
import sys
import cv2
import time
import random
import numpy as np
from argparse import ArgumentParser,Namespace
from anomalib.post_processing import Visualizer
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

# initialize
args = get_args()
model_name = args.model
path = args.path
DIM = args.dim

# visualizer
visualizer = Visualizer(mode="simple",task="segmentation")
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
    # output = visualizer.visualize_image(prediction)
    # output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    # path = os.path.join(image_path,'top_left_infer.jpg')
    # cv2.imwrite(path,output)

    # predict top left
    prediction = inferencer.predict(image=top_right)
    # output = visualizer.visualize_image(prediction)
    # output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    # path = os.path.join(image_path,'top_right_infer.jpg')
    # cv2.imwrite(path,output)

    end_time = time.time() - start_time
    print("Inference timing consumption:",end_time)
