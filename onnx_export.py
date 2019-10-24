 
import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import numpy as np

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
#img = Image.open(opt.image)
#img_to_tensor = ToTensor()
#input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)
pixels = 32
img = np.random.rand(pixels,pixels,3)
input = torch.from_numpy(img).view(1,3,pixels,pixels).float().to(device)
#print('input image size {:d}x{:d}'.format(img.size[0], img.size[1]))
print("input size is..", input.shape)

# load the model
from models import *
model = ResNet18().to(device)

# this may not coexist with onnx.
#from fp16util import network_to_half
#model = network_to_half(model)

# DataParallel does not coexist with onnx.

print(model)
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['net'])

print("model set!")

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, args.model_out, verbose=True, input_names=input_names, output_names=output_names, opset_version=9)
print('model exported to {:s}'.format(args.model_out))
