 
import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from torch2trt import torch2trt

import time

# exporter settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='res18', help="set model checkpoint path")
parser.add_argument('--model_out', type=str, default='resnet18.onnx')
#parser.add_argument('--image', type=str, required=True, help='input image to use')

args = parser.parse_args() 
print(args)


# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))


# load the image
pixels = 32
img = np.random.rand(pixels,pixels,3)
input = torch.from_numpy(img).view(1,3,pixels,pixels).float().to(device)
print("input size is..", input.shape)

# load the model
from models import *
model = ResNet18()
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['net'])

# set to eval and send to gpu
model = model.eval().to(device)
print("model set!")

pytorch_time = []
batch = 32
for x in range(100):
    
    img = np.random.rand(batch,pixels,pixels,3)
    input = torch.from_numpy(img).view(batch,3,pixels,pixels).float().to(device)
    tic = time.time()
    out = model(input)
    toc = time.time()
    pytorch_time.append(toc-tic)
print("pytorch inference took: ", np.mean(np.asarray(pytorch_time)))
print("pytorch FPS is: ", 1/np.mean(np.asarray(pytorch_time))*batch)

# try Pytorch FP16 inference
from fp16util import network_to_half
model2 = network_to_half(model)
pytorch_time = []
for x in range(100):
    
    img = np.random.rand(batch,pixels,pixels,3)
    input = torch.from_numpy(img).view(batch,3,pixels,pixels).half().to(device)
    tic = time.time()
    out = model2(input)
    toc = time.time()
    pytorch_time.append(toc-tic)
print("FP16 pytorch inference took: ", np.mean(np.asarray(pytorch_time)))
print("FP16 pytorch FPS is: ", 1/np.mean(np.asarray(pytorch_time))*batch)

del model2

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to trt...')
tic = time.time()
model_trt = torch2trt(model, [input], max_batch_size=batch)
toc = time.time()
print("conversion completed! took:", toc-tic)

trt_time = []
for x in range(100):
    
    img = np.random.rand(batch,pixels,pixels,3)
    input = torch.from_numpy(img).view(batch,3,pixels,pixels).half().to(device)
    tic = time.time()
    out = model_trt(input)
    toc = time.time()
    trt_time.append(toc-tic)
print("trt inference took: ", np.mean(np.asarray(trt_time)))
print("trt FPS is: ", 1/np.mean(np.asarray(trt_time))*batch)

