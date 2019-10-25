# pytorch-onnx-tensorrt-CIFAR10
Train cifar10 networks and inference with tensorrt.

# how to run tensorRT inference..

1. train model

`python train.py`

2. convert the model to onnx

` python onnx_export.py --model checkpoint/res18-ckpt.t7 `

3. inference

` python inference.py --onnx resnet18.onnx `
