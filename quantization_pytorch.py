
import os
import torch
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as ttf
import math
import os
import matplotlib.pyplot as plt
import time
import os.path as osp
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision.io import read_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx): #self.img_labels['image_path'].values[idx]

        img_path = os.path.join(self.img_dir, self.img_labels['image_path'].values[idx])
        image = read_image(img_path)
        label = self.img_labels['mapped_label'].values[idx]
        return image, label

DATA_DIR = ""
TRAIN_DIR = osp.join(DATA_DIR, "DF20M/")
train_dataset = CustomImageDataset( 'DF20M_train_mapped.csv', TRAIN_DIR)
print(train_dataset.__getitem__(1)[1])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)

TEST_DIR = osp.join(DATA_DIR, "DF20M/")
test_loader = CustomImageDataset( 'DF20M_mapped.csv', TEST_DIR)

test_loader = DataLoader(test_loader, batch_size=64, shuffle=False, num_workers=8)
images, labels = next(iter(train_loader))
print("Train dataset samples ={}, batches = {}".format( train_dataset.__len__(), len(train_loader)))


# Resnet 18 Model class from Pytorch

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding
    
    Args:
        in_planes: number of channels in input image
        out_planes: number of channels produced by convolution
        stride: stride of the convolution. Default: 1
        groups: Number of blocked connections from input channels to output channels. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        
    Returns:
        Convoluted layer of kernel size=3, with specified out_planes
    
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    
    Args:
        in_planes: number of channels in input image
        out_planes: number of channels produced by convolution
        stride: stride of the convolution. Default: 1
        
    Returns:
        Convoluted layer of kernel size=1, with specified out_planes
        
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, quantize=False):
        super(BasicBlock, self).__init__()
        # initialize self quantize 
        self.quantize = quantize
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # added quantized float functional 
        self.skip_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.quantize:
            out = self.skip_add.add(out, identity)
        else:
            out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=182, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, quantize=False):
        super(ResNet, self).__init__()
        #initialize quantize attribute
        self.quantize = quantize
        num_channels = 3
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # start quantize at the beginning of the model 
        self.quant = torch.quantization.QuantStub()
        # endthe model with dequantization of the model
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, quantize=self.quantize))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, quantize=self.quantize))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # quantize the input
        if self.quantize:
            x = self.quant(x)
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # dequantize output 
        if self.quantize:
            x = self.dequant(x)

        return x

    def forward(self, x):
         
        return self._forward_impl(x)

def test(model, device, test_loader, quantize=False, fbgemm=False, all_layers=True, is_relu=False):
    
    model.eval()
    avg_elapsed = []
    # begin qauntization if True
    if quantize:
        if all_layers:
            if is_relu:
                layers_to_fuse = [['conv1', 'bn1', 'relu'],
                        ['layer1.0.conv1', 'layer1.0.bn1','layer1.0.relu' ],
                        ['layer1.0.conv2', 'layer1.0.bn2'],
                        ['layer1.1.conv1', 'layer1.1.bn1','layer1.1.relu' ],
                        ['layer1.1.conv2', 'layer1.1.bn2'],
                        ['layer2.0.conv1', 'layer2.0.bn1','layer2.0.relu' ],
                        ['layer2.0.conv2', 'layer2.0.bn2'],
                        ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
                        ['layer2.1.conv1', 'layer2.1.bn1','layer2.1.relu' ],
                        ['layer2.1.conv2', 'layer2.1.bn2'],
                        ['layer3.0.conv1', 'layer3.0.bn1','layer3.0.relu' ],
                        ['layer3.0.conv2', 'layer3.0.bn2'],
                        ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
                        ['layer3.1.conv1', 'layer3.1.bn1','layer3.1.relu' ],
                        ['layer3.1.conv2', 'layer3.1.bn2'],
                        ['layer4.0.conv1', 'layer4.0.bn1','layer4.0.relu' ],
                        ['layer4.0.conv2', 'layer4.0.bn2'],
                        ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
                        ['layer4.1.conv1', 'layer4.1.bn1','layer4.1.relu' ],
                        ['layer4.1.conv2', 'layer4.1.bn2']]
            else:
                layers_to_fuse = [['conv1', 'bn1'],
                        ['layer1.0.conv1', 'layer1.0.bn1'],
                        ['layer1.0.conv2', 'layer1.0.bn2'],
                        ['layer1.1.conv1', 'layer1.1.bn1'],
                        ['layer1.1.conv2', 'layer1.1.bn2'],
                        ['layer2.0.conv1', 'layer2.0.bn1'],
                        ['layer2.0.conv2', 'layer2.0.bn2'],
                        ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
                        ['layer2.1.conv1', 'layer2.1.bn1'],
                        ['layer2.1.conv2', 'layer2.1.bn2'],
                        ['layer3.0.conv1', 'layer3.0.bn1'],
                        ['layer3.0.conv2', 'layer3.0.bn2'],
                        ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
                        ['layer3.1.conv1', 'layer3.1.bn1'],
                        ['layer3.1.conv2', 'layer3.1.bn2'],
                        ['layer4.0.conv1', 'layer4.0.bn1'],
                        ['layer4.0.conv2', 'layer4.0.bn2'],
                        ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
                        ['layer4.1.conv1', 'layer4.1.bn1'],
                        ['layer4.1.conv2', 'layer4.1.bn2']]
                        
                
        else:
            if is_relu:
            # quantize one conv layer
                layers_to_fuse = [['conv1', 'bn1', 'relu'],
                        ['layer1.0.conv1', 'layer1.0.bn1','layer1.0.relu' ],
                        ['layer1.0.conv2', 'layer1.0.bn2'],
                        ['layer1.1.conv1', 'layer1.1.bn1','layer1.1.relu' ],
                        ['layer1.1.conv2', 'layer1.1.bn2']]
            else:
                layers_to_fuse = [['conv1', 'bn1'],
                        ['layer1.0.conv1', 'layer1.0.bn1'],
                        ['layer1.0.conv2', 'layer1.0.bn2'],
                        ['layer1.1.conv1', 'layer1.1.bn1'],
                        ['layer1.1.conv2', 'layer1.1.bn2']]


        # we attached the global qconfig, in this case we used fbgemm for x86 CPU
        if fbgemm:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            model.qconfig = torch.quantization.default_qconfig

        # We fused targetted layers
        model = torch.quantization.fuse_modules(model, layers_to_fuse)
        # we prepared the model for static quantization heere where we inserted obervers for calibration
        torch.quantization.prepare(model, inplace=True)
        # take dropout  layers for inference
        model.eval()

        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                data = data.to(torch.float)
                target = target.to(torch.float)
                #we calibrated the model with representative data 
                model(data)
                
                batch_bar.update()
        # we converted the observed model to  a quantized model here 
        # where we quantized the weights compute and stored weight scale & bias
        torch.quantization.convert(model, inplace=True)
        batch_bar.close()

    num_correct = 0
    with torch.no_grad():
        batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            data = data.to(torch.float)
            label = label.to(torch.float)
            # start time
            st = time.time()
            outputs = model(data)
            # end time
            et = time.time()
            elapsed_time = (et - st) * 1000
            avg_elapsed.append(elapsed_time)
            pred = torch.argmax(outputs, dim=-1) # get the index of the max probability
            # get correct predictions
            num_correct += int((pred == label).sum())
            batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((batch_idx + 1) * 64)))
            batch_bar.update()
    print("===========Quantization Results================")
    torch.save(model.state_dict(), "model_size.pth")
    print('Model Size (MB):', os.path.getsize("model_size.pth")/1e6)
    os.remove('model_size.pth')
    average = sum(avg_elapsed)/len(avg_elapsed)
    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(num_correct, len(test_loader.dataset),
        100. * num_correct / len(test_loader.dataset)))
    print('Elapsed inference time = {:0.4f} milliseconds'.format(average))
    print("===============================================")


device = 'cpu'
model = ResNet(num_classes=182, quantize=True)
model.load_state_dict(torch.load("DF20M-ResNet18_best_accuracy.pth"))
print(model.eval())
# Baseline performance - unquantized model
test(model, device=device, test_loader=test_loader)

# Is relu True
# quantized model all layers
test(model, device=device, test_loader=test_loader, quantize=True,all_layers=True, is_relu=True)
# quantized model only 1 conv layer
test(model, device=device, test_loader=test_loader, quantize=True,all_layers=False, is_relu=True)


# Is relu False
# quantized model all layers
test(model, device=device, test_loader=test_loader, quantize=True,all_layers=True, is_relu=False)
# quantized model only 1 conv layer
test(model, device=device, test_loader=test_loader, quantize=True,all_layers=False, is_relu=False)

