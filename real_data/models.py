
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torch.nn.utils.weight_norm as weightNorm

class WNScaling(nn.Module):
    def __init__(self, input_features, output_features):
        super(WNScaling, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.scaling = (math.sqrt(2.* input_features/ output_features))

    def forward(self, input):
        return input* self.scaling




class MLPNet(nn.Module):
    def __init__(self, nhiddens=[500, 500], dropout=0., normalization='bn', init=''):
        super(MLPNet, self).__init__()
        self.layers = []

        if normalization=='wn':
            self.layers += [weightNorm(nn.Linear(28*28, nhiddens[0], bias=False))]
            if init=='proposed':
                self.layers.append(WNScaling(28*28,nhiddens[0]))
        if normalization=='bn':
            self.layers += [nn.Linear(28*28, nhiddens[0]), nn.BatchNorm1d(nhiddens[0])]
        else:
            self.layers += [nn.Sequential()]
        self.layers += [nn.ReLU()]

        for l in range(len(nhiddens)-1):
            if normalization=='wn':
                self.layers += [weightNorm(nn.Linear(nhiddens[l], nhiddens[l+1], bias=False))]
                if init=='proposed':
                    self.layers.append(WNScaling(nhiddens[l], nhiddens[l+1]))
            if normalization=='bn':
                self.layers += [nn.Linear(nhiddens[l], nhiddens[l+1]), nn.BatchNorm1d(nhiddens[0])]
            else:
                self.layers += [nn.Sequential()]
            self.layers += [nn.ReLU()]

        self.feat = nn.Sequential(*self.layers)
        self.clf = nn.Linear(nhiddens[-1], 10)
        
    def forward(self, x, mask1=None, mask2=None):
        x = x.view(-1, 28*28)
        x = self.feat(x)
        x = (self.clf(x))
        return x
    def name(self):
        return 'mlpnet'
    
    
class resblock(nn.Module):

    def __init__(self, depth, channels, stride=1, dropout=0., normalization='', nresblocks=1., init=''):
        super(resblock, self).__init__()
        self.bn = normalization=='bn'
        self.depth = depth
        self. channels = channels
        if init=='proposed':
            self.nresblocks = nresblocks
        else:
            self.nresblocks = 1.

        if self.bn:
            self.bn1 = nn.BatchNorm2d(depth)
        if normalization=='wn':
            conv1 = [weightNorm(nn.Conv2d(depth, channels, kernel_size=3, stride=stride, padding=1, bias=False))]
            if init=='proposed':
                conv1.append(WNScaling(depth*3*3,channels))
            self.conv1 = nn.Sequential(*conv1)
        else:
            self.conv1 = (nn.Conv2d(depth, channels, kernel_size=3, stride=stride, padding=1, bias=False))


        if self.bn:
            self.bn2 = nn.BatchNorm2d(channels)
        if normalization=='wn':
            conv2 = [weightNorm(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))]
            if init=='proposed':
                conv2.append(WNScaling(channels*3*3,channels))
            self.conv2 = nn.Sequential(*conv2)
        else:
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride > 1:
            layers = []
            if normalization=='wn':
                layers += [weightNorm(nn.Conv2d(depth, channels, kernel_size=1, stride=stride, padding=0, bias=False))]
                if init=='proposed':
                    layers.append(WNScaling(depth*1*1,channels))
            else:
                layers += [nn.Conv2d(depth, channels, kernel_size=1, stride=stride, padding=0, bias=False)]
            self.shortcut = nn.Sequential(*layers)
            
        self.dropout = dropout

    def forward(self, x):
#         print 'input shape: ', x.size()
#         print 'depth, channels: ', self.depth, self.channels
        if self.bn:
            out = F.relu(self.bn1(x))
            out = F.relu(self.bn2(self.conv1(out)))
        else:
            out = F.relu((x))
            out = F.relu((self.conv1(out)))
        if self.dropout>0:
            out = nn.Dropout(self.dropout)(out)
        out = (1./self.nresblocks)*self.conv2(out)
        out += 1.*self.shortcut(x)
        return out



class ResNet(nn.Module):
    def __init__(self, n=9, nb_filters=16, num_classes=10, dropout=0., normalization='', dataset=None, init=''): # n=9->Resnet-56
        super(ResNet, self).__init__()
        nstage = 3
        self.dataset=dataset
        self.layers = []
        
        self.num_classes = num_classes
        if normalization=='wn':
            conv1 = weightNorm(nn.Conv2d(3, nb_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.layers.append(conv1)
            if init=='proposed':
                self.layers.append(WNScaling(3*3*3,nb_filters))
        else:
            conv1 = (nn.Conv2d(3, nb_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.layers.append(conv1)
            if normalization=='bn':
                layer = nn.BatchNorm2d(nb_filters)
                self.layers.append(layer)
        
        nb_filters_prev = nb_filters_cur = nb_filters
        
        for stage in range(nstage):
            nb_filters_cur = (2 ** stage) * nb_filters
            for i in range(n):
                subsample = 1 if (i > 0 or stage == 0) else 2
                layer = resblock(nb_filters_prev, nb_filters_cur, subsample, dropout=dropout, normalization=normalization, nresblocks = nstage*n, init=init)
                self.layers.append(layer)
                nb_filters_prev = nb_filters_cur
        
        if normalization=='bn':
            layer = nn.BatchNorm2d(nb_filters_cur)
            self.layers.append(layer)
        
        self.pre_clf = nn.Sequential(*self.layers)
        
        self.fc = nn.Linear(nb_filters_cur, self.num_classes) # assuming the last conv hidden state is of size (N, nb_filters_cur, 1, 1) 
        
        
        
    def forward(self, x):
        out = x
#         for layer in self.layers:
#             out = layer(out)
        
        out = self.pre_clf(out)
        out = F.relu(out)
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            out = nn.AvgPool2d(8,8)(out)
        elif self.dataset=='imagenet':
            out = nn.AvgPool2d(16,16)(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        
        return out



# Resnet nomenclature: 6n+2 = 3x2xn + 2; 3 stages, each with n number of resblocks containing 2 conv layers each, and finally 2 non-res conv layers
def ResNet56(dropout=0., normalization='bn', num_classes=10, dataset='cifar10', init=''):
    return ResNet(n=9, nb_filters=16, num_classes=num_classes, dropout=dropout, normalization=normalization, dataset=dataset, init=init)
def ResNet_model(dropout=0., normalization='bn', num_classes=10, dataset='cifar10', n=9, init=''):
    return ResNet(n=n, nb_filters=16, num_classes=num_classes, dropout=dropout, normalization=normalization, dataset=dataset, init=init)


