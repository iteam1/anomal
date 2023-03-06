import os
import cv2
import numpy as np
from anomalib.deploy import TorchInferencer,OpenVINOInferencer

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

config_path = 'models/cflow/config.yaml'
model_path = 'models/cflow/weights/model.ckpt'
inferencer = TorchInferencer(config=config_path,model_source=model_path,device ='auto')

if __name__ == "__main__":
    # read image
    image = cv2.imread('datasets/laptop/test/crack/000.png')
    print(image.shape)
    # predict
    prediction = inferencer.predict(image=image)
    output = visualizer.visualize_image(prediction)
    #print(output.shape)
    cv2.imshow('output',output)
    k = cv2.waitkey()
    cv2.imwrite('output.png',output)
    cv2.destroyAllWindows()