import os
import sys
import cv2
import time
import random
import numpy as np
from anomalib.post_processing import Visualizer
from anomalib.deploy import TorchInferencer, OpenVINOInferencer
#from anomalib.post_processing.visualizer import Visualizer
#from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
#from anomalib.deploy.inferencers.openvino_inferencer import OpenVINOInferencer

model_name = sys.argv[1]
path = 'samples/crack'
DIM = 256

# visualizer
visualizer = Visualizer(mode="simple",task="segmentation")

# config_path = 'model/dfm/mvtec/laptop/run/config.yaml'
# weight_path = 'model/dfm/mvtec/laptop/run/openvino/model.onnx'
# meta_data_path = 'models/dfm/mvtec/laptop/run/openvino/meta_data.json'
# device = 'CPU' #["CPU", "GPU", "VPU"]
# inferencer = OpenVINOInferencer(
#     config = config_path,
#     path = weight_path,
#     meta_data_path = meta_data_path,
#     device = device
# )

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
    output = visualizer.visualize_image(prediction)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    cv2.imwrite('top_left_infer.jpg',output)

    # predict top left
    prediction = inferencer.predict(image=top_right)
    output = visualizer.visualize_image(prediction)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    cv2.imwrite('infer_top_right_infer.jpg',output)

    end_time = time.time() - start_time
    print("Inference timing consumption:",end_time)
