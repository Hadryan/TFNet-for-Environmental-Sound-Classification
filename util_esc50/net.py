import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        
    def show(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return x
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    
class TFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(TFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                             out_channels=out_channels,
                                 kernel_size=(3, 3), stride=(1, 1),
                             padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.alpha = nn.Parameter(torch.cuda.FloatTensor([.1, .1, .1]))
        self.bnx = nn.BatchNorm2d(1)
        self.bny = nn.BatchNorm2d(1)
        self.bnz = nn.BatchNorm2d(out_channels)
        self.bna = nn.BatchNorm2d(out_channels)
        self.bnb = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=1,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=False)
        self.conv4 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=1,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=False)
        if out_channels == 64:
            self.globalAvgPool2 = nn.AvgPool2d((250,1), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((1,40), stride=1)
            self.globalMaxPool2 = nn.MaxPool2d((1,64), stride=1)
            self.globalMaxPool3 = nn.MaxPool2d((64,1), stride=1)
            self.fc1 = nn.Linear(in_features=40, out_features=40)
            self.fc2 = nn.Linear(in_features=250, out_features=250)
            
        elif out_channels == 128:
            self.globalAvgPool2 = nn.AvgPool2d((125,1), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((1,20), stride=1)
            self.globalMaxPool2 = nn.MaxPool2d((1,128), stride=1)
            self.globalMaxPool3 = nn.MaxPool2d((128,1), stride=1)
            self.fc1 = nn.Linear(in_features=20, out_features=20)
            self.fc2 = nn.Linear(in_features=125, out_features=125)
        elif out_channels == 256:
            self.globalAvgPool2 = nn.AvgPool2d((62,1), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((1,10), stride=1)
            self.globalMaxPool2 = nn.MaxPool2d((1,128), stride=1)
            self.globalMaxPool3 = nn.MaxPool2d((128,1), stride=1)
            self.fc1 = nn.Linear(in_features=10, out_features=10)
            self.fc2 = nn.Linear(in_features=62, out_features=62)
        elif out_channels == 512:
            self.globalAvgPool2 = nn.AvgPool2d((31,1), stride=1)
            self.globalAvgPool3 = nn.AvgPool2d((1,5), stride=1)
            self.globalMaxPool2 = nn.MaxPool2d((1,128), stride=1)
            self.globalMaxPool3 = nn.MaxPool2d((128,1), stride=1)
            self.fc1 = nn.Linear(in_features=5, out_features=5)
            self.fc2 = nn.Linear(in_features=31, out_features=31)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.downsample = conv1x1(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    
    def show(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        out1 = x.clone()
        res = x.clone()
        y = x.clone()
        y = self.bnx(self.relu(self.conv3(y)))
        out6 = y.clone()
        res_2 = x.clone()
        z = x.clone()
        z = self.bny(self.relu(self.conv4(z)))
        res_3 = x.clone()
        out7 = z.clone()
        h = x.clone()
        
        res_2 = res_2.transpose(1,3)
        y = y.transpose(1,3)
        y = self.globalAvgPool2(y)
        y = y.view(y.size(0), -1)
        y = self.sigmoid(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * res_2
        y = y.transpose(1,3)
        y = self.bna(y)
        out2=y.clone()
        res_3 = res_3.transpose(1,2)
        z = z.transpose(1,2)
        z = self.globalAvgPool3(z)
        z = z.view(z.size(0), -1)
        z = self.sigmoid(z)
        z = z.view(z.size(0), z.size(1), 1, 1)
        z = z * res_3
        z = z.transpose(1,2)
        z = self.bnb(z)
        out3 = z.clone()
        so_alpha = F.softmax(self.alpha,dim=0)
        x = so_alpha[0]*h + so_alpha[1]*y + so_alpha[2]*z
        x = self.relu(x)
        out4 = x.clone()
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        out5 = x.clone()
        out1 = torch.mean(out1, dim=1)
        out2 = torch.mean(out2, dim=1)
        out3 = torch.mean(out3, dim=1)
        out4 = torch.mean(out4, dim=1)
        out5 = torch.mean(out5, dim=1)
        return out1, out2, out3, out4, out5, out6, out7
    
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        res = x.clone()
        y = x.clone()
        y = self.bnx(self.relu(self.conv3(y)))
        res_2 = x.clone()
        z = x.clone()
        z = self.bny(self.relu(self.conv4(z)))
        res_3 = x.clone()
        
        h = x.clone()
        
        res_2 = res_2.transpose(1,3)
        y = y.transpose(1,3)
        y = self.globalAvgPool2(y)
        y = y.view(y.size(0), -1)
        y = self.sigmoid(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * res_2
        y = y.transpose(1,3)
        y = self.bna(y)
        res_3 = res_3.transpose(1,2)
        z = z.transpose(1,2)
        z = self.globalAvgPool3(z)
        z = z.view(z.size(0), -1)
        z = self.sigmoid(z)
        z = z.view(z.size(0), z.size(1), 1, 1)
        z = z * res_3
        z = z.transpose(1,2)
        z = self.bnb(z)
        so_alpha = F.softmax(self.alpha,dim=0)
        x = so_alpha[0]*h + so_alpha[1]*y + so_alpha[2]*z
        x = self.relu(x)
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
class TFNet(nn.Module):
    
    def __init__(self, classes_num=10, activation='logsoftmax'):
        super(TFNet, self).__init__()

        self.activation = activation

        self.tfblock1 = TFBlock(in_channels=1, out_channels=64)
        self.tfblock2 = TFBlock(in_channels=64, out_channels=128)
        self.tfblock3 = TFBlock(in_channels=128, out_channels=256)
        self.tfblock4 = TFBlock(in_channels=256, out_channels=512)
        self.fc = nn.Linear(512, classes_num, bias=True)

        
    def show(self, input):
       
        x = input[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        out1, out2, out3, out4, out5, out6, out7 = self.conv_block1.show(x)
        x = self.tfblock1(x, pool_size=(2, 2), pool_type='avg')
        x1 = torch.mean(x, dim=1)
        x = self.tfblock2(x, pool_size=(2, 2), pool_type='avg')
        x2 = torch.mean(x, dim=1)
        x = self.tfblock3(x, pool_size=(2, 2), pool_type='avg')
        x3 = torch.mean(x, dim=1)
        x = self.tfblock4(x, pool_size=(2, 2), pool_type='avg')
        x4 = torch.mean(x, dim=1)
        return x1, x2, x3, x4, out1, out2, out3, out4, out5, out6, out7
    
    def forward(self, input):
        '''
        Input: (batch_size, seq_number, times_steps, freq_bins)'''
        
        x = input[:, 0 , : , :]
        x = x[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.tfblock1(x, pool_size=(2, 2), pool_type='avg')
        x = self.tfblock2(x, pool_size=(2, 2), pool_type='avg')
        x = self.tfblock3(x, pool_size=(2, 2), pool_type='avg')
        x = self.tfblock4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc(x)
        
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output  

    
class Cnn(nn.Module):
    
    def __init__(self, classes_num=50, activation='logsoftmax'):
        super(Cnn, self).__init__()

        self.activation = activation
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, classes_num, bias=True)

    def forward(self, input):
        '''
        Input: (batch_size, seq_number, times_steps, freq_bins)'''

        x = input[:, 0, :, :]
        x = x[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)

        return output 
