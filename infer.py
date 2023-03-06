import os
import cv2
import numpy as np
#from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
from anomalib.deploy.inferencers.openvino_inferencer import OpenVINOInferencer

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

# config_path = 'models/fastflow/mvtec/laptop/run/config.yaml'
# model_path = 'models/fastflow/mvtec/laptop/run/weights/model.ckpt'
# inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

if __name__ == "__main__":
    pass