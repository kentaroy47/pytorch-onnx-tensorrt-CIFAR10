# pytorch-onnx-tensorrt-CIFAR10
Train cifar10 networks and inference with tensorrt.

** The Tensor RT inference is about 5 times faster than Pytorch on Jetson Nano :) **

Also note that FP16 pytorch models are quite fast too.

# how to run tensorRT inference..

1. train model

`python train.py`

2. convert the model to onnx

` python onnx_export.py --model checkpoint/res18-ckpt.t7 `

3. Tensor RT inference

` python3 trt_convert.py --model checkpoint/res18-ckpt.t7 `


Device: Jetson Nano

Network: Resnet 18

Pytorch FP32: 46ms

Pytorch FP16: 16ms

TensorRT: 4.2ms

```
pytorch inference took:  0.046734986305236814
FP16 pytorch inference took:  0.01620328426361084
exporting model to trt...
conversion completed! took: 20.77101159095764
trt inference took:  0.004269411563873291

```
