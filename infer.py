import os
import cv2
import time
import numpy as np
from anomalib.post_processing import Visualizer
from anomalib.deploy import TorchInferencer
#from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
#from anomalib.deploy.inferencers.openvino_inferencer import OpenVINOInferencer

visualizer = Visualizer(mode="full",task="segmentation")

# config_path = 'models/patchcore/mvtec/hazelnut/run/config.yaml'
# weight_path = 'models/patchcore/mvtec/hazelnut/run/openvino/model.xml' # yml,onnx
# meta_data_path = 'models/patchcore/mvtec/hazelnut/run/openvino/meta_data.json'
# device = 'CPU' #["CPU", "GPU", "VPU"]
#
# inferencer = OpenVINOInferencer(
#     config = config_path,
#     path = weight_path,
#     meta_data_path = meta_data_path,
#     device = device
# )

config_path = 'models/fastflow/mvtec/laptop/run/config.yaml'
model_path = 'models/fastflow/mvtec/laptop/run/weights/model.ckpt'
inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

if __name__ == "__main__":
    # Start counting time
    start_time = time.time()
    image = cv2.imread('datasets/laptop/test/crack/002.png')
    # predict
    prediction = inferencer.predict(image=image)
    output = visualizer.visualize_image(prediction)
    # print(output.shape)
    cv2.imwrite('output.jpg',output)
    end_time = time.time() - start_time
    print("Inference timing consumption:",end_time)