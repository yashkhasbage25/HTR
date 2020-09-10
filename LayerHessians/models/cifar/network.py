import re
import math
#import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from collections import OrderedDict
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

# from IPython import embed

class Network:
    def construct(self, net, obj):
        targetClass = getattr(self, net)
        instance = targetClass(obj)
        return instance
    
    ###########################################################################
    ##############################      VGG      ##############################
    ###########################################################################
        
    class VGG(nn.Module):
        def __init__(self, obj, net_type, batch_norm):
            super(Network.VGG, self).__init__()
            
            self.features = self.make_layers(obj.input_ch, Network.cfg[net_type], batch_norm=batch_norm)
            
            num_strides = sum([layer == 'M' for layer in Network.cfg[net_type]])
            kernel_numel = int((obj.padded_im_size / (2**num_strides))**2)

            relu1 = nn.ReLU(inplace=False)
            relu2 = nn.ReLU(inplace=False)
            
            lin1 = nn.Linear(512 * kernel_numel, 4096, bias=False)
            lin2 = nn.Linear(4096, 4096, bias=False)
            lin3 = nn.Linear(4096, 1000)
            
            bn1 = nn.BatchNorm1d(4096)
            bn2 = nn.BatchNorm1d(4096)
            
            self.classifier = nn.Sequential(
                lin1,
                bn1,
                relu1,
                lin2,
                bn2,
                relu2,
                lin3
            )
            
            self._initialize_weights()
            
            mod = list(self.classifier.children())
            mod.pop()
            
            lin4 = torch.nn.Linear(4096, obj.num_classes)
            
            mod.append(lin4)
            self.classifier = torch.nn.Sequential(*mod)
            self.classifier[-1].weight.data.normal_(0, 0.01)
            self.classifier[-1].bias.data.zero_()
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    try:
                        m.bias.data.zero_()
                    except:
                        pass
                    
        def make_layers(self, input_ch, cfg, batch_norm=False):
            layers  = []

            in_channels = input_ch
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    relu = nn.ReLU(inplace=False)

                    if batch_norm:
                        bn = nn.BatchNorm2d(v)

                        layers += [conv2d, bn, relu]
                    else:
                        layers += [conv2d, relu]
                    in_channels = v
            return nn.Sequential(*layers)
    
    
    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    
    class VGG11(VGG):
        def __init__(self, obj):
            super(Network.VGG11, self).__init__(obj, 'A', False)
            
    class VGG11_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG11_bn, self).__init__(obj, 'A', True)

    class VGG13(VGG):
        def __init__(self, obj):
            super(Network.VGG13, self).__init__(obj, 'B', False)
            
    class VGG13_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG13_bn, self).__init__(obj, 'B', True)
        
    class VGG16(VGG):
        def __init__(self, obj):
            super(Network.VGG16, self).__init__(obj, 'D', False)
            
    class VGG16_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG16_bn, self).__init__(obj, 'D', True)

    class VGG19(VGG):
        def __init__(self, obj):
            super(Network.VGG19, self).__init__(obj, 'E', False)
            
    class VGG19_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG19_bn, self).__init__(obj, 'E', True)
    
    ###########################################################################
    #############################      ResNet      ############################
    ###########################################################################
    
    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    
    @staticmethod
    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
    
    class BasicBlock(nn.Module):
        expansion = 1
    
        def __init__(self, inplanes, planes, last, stride=1, downsample=None):
            super(Network.BasicBlock, self).__init__()
            self.conv1 = Network.conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU(inplace=True)
            
            self.conv2 = Network.conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            if last:
                self.relu2 = nn.ReLU(inplace=False)
            else:
                self.relu2 = nn.ReLU(inplace=True)
            
            self.downsample = downsample
            self.stride = stride
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu2(out)
    
            return out
    
    
    class Bottleneck(nn.Module):
        expansion = 4
    
        def __init__(self, inplanes, planes, last, stride=1, downsample=None):
            super(Network.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            if last:
                self.relu3 = nn.ReLU(inplace=False)
            else:
                self.relu3 = nn.ReLU(inplace=True)
                
            self.downsample = downsample
            self.stride = stride
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
    
            out = self.conv3(out)
            out = self.bn3(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu3(out)
    
            return out
    
    class ResNet(nn.Module):
    
        def __init__(self, obj, block, layers):
            self.obj = obj
            self.inplanes = 64
            super(Network.ResNet, self).__init__()
            
            if obj.resnet_type == 'big':
                self.conv1 = nn.Conv2d(obj.input_ch, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
            elif obj.resnet_type == 'small':
                self.conv1 = Network.conv3x3(obj.input_ch, 64)
                
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
            
            if obj.resnet_type == 'big':
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            
            if obj.resnet_type == 'big':
                num_strides = 5
            elif obj.resnet_type == 'small':
                num_strides = 3
            
            kernel_sz = int(obj.padded_im_size / (2**num_strides))
            self.avgpool = nn.AvgPool2d(kernel_sz, stride=1)
            self.fc = nn.Linear(512 * block.expansion, obj.num_classes)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
    
            layers = []
            layers.append(block(self.inplanes, planes, False, stride, downsample))
            
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                
                last = (planes == 512) & (i == blocks-1)
                
                cur_layers = block(self.inplanes, planes, last)                
                layers.append(cur_layers)
    
            return nn.Sequential(*layers)
        
        def forward(self, x):
                        
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            if hasattr(self, 'maxpool'):
                x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
            return x
            
    class ResNet18(ResNet):
        def __init__(self, obj):
            super(Network.ResNet18, self).__init__(obj, Network.BasicBlock, [2, 2, 2, 2])
    
    class ResNet34(ResNet):
        def __init__(self, obj):
            super(Network.ResNet34, self).__init__(obj, Network.BasicBlock, [3, 4, 6, 3])
    
    class ResNet50(ResNet):
        def __init__(self, obj):
            super(Network.ResNet50, self).__init__(obj, Network.Bottleneck, [3, 4, 6, 3])
    
    class ResNet101(ResNet):
        def __init__(self, obj):
            super(Network.ResNet101, self).__init__(obj, Network.Bottleneck, [3, 4, 23, 3])

    class ResNet152(ResNet):
        def __init__(self, obj):
            super(Network.ResNet152, self).__init__(obj, Network.Bottleneck, [3, 8, 36, 3])

    ###########################################################################
    ###########################      DenseNet      ############################
    ###########################################################################

    class DenseNetBasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(Network.DenseNetBasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
            
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            return torch.cat([x, out], 1)
    
    class BottleneckBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(Network.BottleneckBlock, self).__init__()
            inter_planes = out_planes * 4
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(inter_planes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
            
        def forward(self, x):
            out = self.conv1(self.relu1(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            out = self.conv2(self.relu2(self.bn2(out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return torch.cat([x, out], 1)
    
    class TransitionBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(Network.TransitionBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.droprate = dropRate
            
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return F.avg_pool2d(out, 2)
        
    class DenseBlock(nn.Module):
        def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
            super(Network.DenseBlock, self).__init__()
            self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
            
        def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
            layers = []
            self.relus = []
            self.convs = []
            for i in range(nb_layers):
                b = block(in_planes+i*growth_rate, growth_rate, dropRate)
                layers.append(b)
            return nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layer(x)
    
    class DenseNet3(nn.Module):
        def __init__(self, obj, depth, growth_rate=12,
                     reduction=0.5, bottleneck=True, dropRate=0.0):
            super(Network.DenseNet3, self).__init__()
            in_planes = 2 * growth_rate
            n = (depth - 4) / 3
            if bottleneck == True:
                n = n/2
                block = Network.BottleneckBlock
            else:
                block = Network.DenseNetBasicBlock
            n = int(n)
            
            # 1st conv before any dense block
            self.conv1 = nn.Conv2d(obj.input_ch, in_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            
            # 1st block
            self.block1 = Network.DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            self.trans1 = Network.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes*reduction))
            
            # 2nd block
            self.block2 = Network.DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            self.trans2 = Network.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes*reduction))
            
            # 3rd block
            self.block3 = Network.DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(in_planes, obj.num_classes)
            self.in_planes = in_planes
            
            kernel_sz = int(obj.padded_im_size / (2**2))    
            self.avgpool = nn.AvgPool2d(kernel_size=kernel_sz, stride=1)
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
                    
        def forward(self, x):
            out = self.conv1(x)
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = self.avgpool(out)
            out = out.view(-1, self.in_planes)
            return self.fc(out)
        
    class DenseNet3_40(DenseNet3):
        def __init__(self, obj):
            super(Network.DenseNet3_40, self).__init__(obj, depth=40, growth_rate=12, reduction=1, bottleneck=False, dropRate=0.0)
            
    ########################################################################################
    ################################# MOBILENET ############################################
    ########################################################################################

    class MobileNetBlock(nn.Module):
        '''Depthwise conv + Pointwise conv'''
        def __init__(self, in_planes, out_planes, stride=1):
            super(Network.MobileNetBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            return out


    class MobileNet(nn.Module):
        # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
        cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

        def __init__(self, obj): #, num_classes=10, in_channels=3):
            self.obj = obj
            super(Network.MobileNet, self).__init__()
            self.conv1 = nn.Conv2d(obj.input_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.layers = self._make_layers(in_planes=32)
            self.linear = nn.Linear(1024, obj.num_classes)

        def _make_layers(self, in_planes):
            layers = []
            for x in self.cfg:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Network.MobileNetBlock(in_planes, out_planes, stride))
                in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layers(out)
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    class FullyConnectedBlock(nn.Module):

        def __init__(self, hidden_dim, num_hidden=2, skip=False):
            
            super(Network.FullyConnectedBlock, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_hidden = num_hidden
            self.skip = skip
            self.lin = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))

        def forward(self, x):

            out = F.relu(self.lin(x))
            if self.skip:
                out = out + x
            return out


    class FullyConnected(nn.Module):

        def __init__(self, obj, num_hidden, skip):

            super(Network.FullyConnected, self).__init__()
            self.input_dim = obj.input_ch * obj.padded_im_size * obj.padded_im_size
            self.hidden_dim = 500
            self.num_classes = obj.num_classes
            self.num_hidden = num_hidden
            self.skip = skip
            
            self.layers = nn.Sequential()

            self.layers.add_module('init', nn.Linear(self.input_dim, self.hidden_dim))
            for i in range(self.num_hidden):
                self.layers.add_module('hidden%d' % i, Network.FullyConnectedBlock(self.hidden_dim, skip=self.skip))
            self.layers.add_module('last', nn.Linear(self.hidden_dim, self.num_classes))

        def forward(self, x):

            x = x.view(x.size(0), -1)
            x = self.layers(x)

            return x

    def FullyConnected3(self, obj):

        return Network.FullyConnected(obj, num_hidden=1, skip=False)

    def FullyConnected5(self, obj):

        return Network.FullyConnected(obj, num_hidden=2, skip=False)

    def FullyConnected8(self, obj):

        return Network.FullyConnected(obj, num_hidden=3, skip=False)

    def FullyConnectedSkip3(self, obj):

        return Network.FullyConnected(obj, num_hidden=1, skip=True)

    def FullyConnectedSkip5(self, obj):

        return Network.FullyConnected(obj, num_hidden=2, skip=True)

    def FullyConnectedSkip8(self, obj):

        return Network.FullyConnected(obj, num_hidden=3, skip=True)


    class SimpleMNIST(nn.Module):

        def __init__(self, obj):
            super(Network.SimpleMNIST, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
            self.relu2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.lin3 = nn.Linear(64 * 6 * 6, 128)
            self.relu3 = nn.ReLU()
            self.lin4 = nn.Linear(128, obj.num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            x = x.view(-1, 64 * 6 * 6)
            x = self.lin3(x)
            x = self.relu3(x)
            x = self.lin4(x)
            return x

    class LeNet(nn.Module):
        def __init__(self, obj):
            super(Network.LeNet, self).__init__()
            self.conv1 = nn.Conv2d(obj.input_ch, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1   = nn.Linear(self.dummy_forward(torch.randn(2, obj.input_ch, obj.padded_im_size, obj.padded_im_size)).size(1), 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, obj.num_classes)

        def forward(self, x):
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out

        def dummy_forward(self, x):

            with torch.no_grad():
                out = F.relu(self.conv1(x))
                out = F.max_pool2d(out, 2)
                out = F.relu(self.conv2(out))
                out = F.max_pool2d(out, 2)
                out = out.view(out.size(0), -1)
            return out

    class GBN(nn.Module):

        def __init__(self, obj):
            super(Network.GBN, self).__init__()
            self.conv1 = nn.Conv2d(obj.input_ch, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1   = nn.Linear(self.dummy_forward(torch.randn(2, obj.input_ch, obj.padded_im_size, obj.padded_im_size)).size(1), 4)
            self.fc2   = nn.Linear(4, 120)
            self.fc3   = nn.Linear(120, 84)
            self.fc4   = nn.Linear(84, obj.num_classes)

        def forward(self, x):
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = self.fc4(out)
            return out

        def dummy_forward(self, x):

            with torch.no_grad():
                out = F.relu(self.conv1(x))
                out = F.max_pool2d(out, 2)
                out = F.relu(self.conv2(out))
                out = F.max_pool2d(out, 2)
                out = out.view(out.size(0), -1)
            return out

    class SimpleSVHN(nn.Module):

        def __init__(self, obj):
            super(Network.SimpleSVHN, self).__init__()
            self.features = nn.Sequential(
                # block 1
                nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2),
                # block 2
                nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2),
                # block 3
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2),
                # block 4
                nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=160),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2),
                # block 5
                nn.Conv2d(in_channels=160, out_channels=192,  kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=192),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2),
                # block 6
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=192),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2),
                # block 7
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=192),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2),
                # block 8
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=192),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            )

            self.classifier = nn.Sequential(
                nn.Linear(192 * 5 * 5, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU()
            )

            # self._digit_length = nn.Sequential(nn.Linear(1000, 7))
            self._digit1 = nn.Sequential(nn.Linear(1000, obj.num_classes))
            # self._digit2 = nn.Sequential(nn.Linear(1000, 11))
            # self._digit3 = nn.Sequential(nn.Linear(1000, 11))
            # self._digit4 = nn.Sequential(nn.Linear(1000, 11))
            # self._digit5 = nn.Sequential(nn.Linear(1000, 11))

        def forward(self, x):

            x = self.features(x)
            # print(x.shape)
            x = x.view(x.size(0), 192 * 5 * 5)
            x = self.classifier(x)

            # length_logits = self._digit_length(x)
            digit1_logits = self._digit1(x)
            # digit2_logits = self._digit2(x)
            # digit3_logits = self._digit3(x)
            # digit4_logits = self._digit4(x)
            # digit5_logits = self._digit5(x)

            # return digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits
            return digit1_logits

    class DeepTanh(nn.Module):

        def __init__(self, obj):

            super(Network.DeepTanh, self).__init__()

            self.hidden_dim = 50
            self.bottleneck_hidden_dim = 3
            self.net = nn.Sequential(
                nn.Linear(obj.input_ch * obj.padded_im_size * obj.padded_im_size, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.bottleneck_hidden_dim),
                
                nn.Tanh(),
                nn.Linear(self.bottleneck_hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, obj.num_classes)
            )

            print('DeepTanh config:')
            print(self.net)

        def forward(self, x):

            out = x.view(x.size(0), -1)
            out = self.net(out)
            return out


    class DeepReLU(nn.Module):

        def __init__(self, obj):

            super(Network.DeepReLU, self).__init__()

            self.hidden_dim = 50
            self.bottleneck_hidden_dim = 3
            self.net = nn.Sequential(
                nn.Linear(obj.input_ch * obj.padded_im_size * obj.padded_im_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.bottleneck_hidden_dim),
                
                nn.ReLU(),
                nn.Linear(self.bottleneck_hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, obj.num_classes)
            )

            print('DeepReLU config:')
            print(self.net)

        def forward(self, x):

            out = x.view(x.size(0), -1)
            out = self.net(out)
            return out

