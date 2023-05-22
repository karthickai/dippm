# DIPPM: a Deep Learning Inference Performance Predictive Model using Graph Neural Networks

## Environment setup
```
# Prerequsite CUDA 11.7

pip install torch==1.13.1 torchvision==0.14.1

pip install torch-geometric==2.2.0

pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.0%2Bpt113cu117-cp310-cp310-linux_x86_64.whl

pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.0%2Bpt113cu117-cp310-cp310-linux_x86_64.whl

pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_sparse-0.6.16%2Bpt113cu117-cp310-cp310-linux_x86_64.whl

pip install pytorch_lightning==1.9.0

pip install networkx apache-tvm timm
```

## Dataset setup
```
git clone https://github.com/karthickai/deeplearning_inference

cd deeplearning_inference

sh dataset.sh
```

## Train the DIPPM
```
python train.py --model_type GraphSAGE --epoch 10
```

### How to use DIPPM
```
import dippm

import torchvision

model = torchvision.models.vgg16(pretrained=True)
model.eval()

dippm.predict(model, batch=8, input="3,244,244", device="A100")

------------------------------------
Predicted Inference Time 7.39 ms
Predicted Power consumption 1511.47 J
Predicted Memory consumption 3083 mb
------------------------------------
```