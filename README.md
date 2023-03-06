# anomal
- clone source: `git clone https://github.com/openvinotoolkit/anomalib.git`
- install `anomalib`: `pip install anomalib==0.4.0`
- install `wandb`: `pip install wandb`
- train `sudo myconda/envs/myenv/bin/python tools/train.py --config configs/cable_config.yaml`
- install training packages `sudo myconda/envs/myenv/bin/pip install -r requirements/openvino.txt`

# packages
inference's requirements:
- pytorch cpu `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`
- opencv `pip install opencv-python`
- openvino `pip install openvino-dev==2021.3.0`

*Note*

[kaggle notebook](https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library)

[source](https://github.com/openvinotoolkit/anomalib/tree/main/anomalib)

[tutorials](https://openvinotoolkit.github.io/anomalib/tutorials/index.html)

[guides](https://openvinotoolkit.github.io/anomalib/how_to_guides/index.html)

[references](https://openvinotoolkit.github.io/anomalib/reference_guide/index.html)

[notebooks](https://github.com/openvinotoolkit/anomalib/tree/main/notebooks)

# references

[Deep-Learning-Based Anomaly Detection with MVTec HALCON](https://www.youtube.com/watch?v=NI6ITCGMhjI)

[Anomalib: Inferences](https://www.youtube.com/watch?v=9KvIS4XgRtg&t=2s)

[pypi v0.4](https://pypi.org/project/anomalib/0.4.0rc2/)

[docs](https://github.com/openvinotoolkit/anomalib/tree/main/docs)

[anomalib](https://github.com/openvinotoolkit/anomalib)

[doc](https://openvinotoolkit.github.io/anomalib/)
