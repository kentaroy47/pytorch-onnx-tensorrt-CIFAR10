# pytorch-onnx-tensorrt-CIFAR10
Train cifar10 networks and inference with tensorrt.

**The Tensor RT inference is about 5 times faster than Pytorch on Jetson Nano :)**

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
running on device cuda:0
input size is.. torch.Size([1, 3, 32, 32])
model set!

pytorch inference took:  0.10171955108642577
pytorch FPS is:  314.5904563893649

FP16 pytorch inference took:  0.02646958112716675
FP16 pytorch FPS is:  1208.9348843966848

exporting model to trt...
conversion completed! took: 16.530654668807983

trt inference took:  0.006784510612487793
trt FPS is:  4716.626124970569


```
