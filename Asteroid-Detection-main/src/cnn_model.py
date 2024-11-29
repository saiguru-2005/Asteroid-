import torch
import torchvision
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18

# Creating a CNN class
class VGG6(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(32*2*2, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.dropout_50(out)
        out = self.fc_2(out)
        return out
    
class VGG6_Input64(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input64, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout_50(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out
    
class VGG6_Input64_Drop20(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input64_Drop20, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.2)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout_50(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out
    
class VGG6_Input64_NoDrop(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input64_NoDrop, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.2)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out
    
class VGG6_Input28(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input28, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(288, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.dropout_50(out)
        out = self.fc_2(out)
        return out
class VGG6_Input100(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input100, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(4608, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.dropout_50(out)
        out = self.fc_2(out)
        return out

class MotionRec_adapted_VGG6(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(32*2*2, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        template = x[0]
        
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.dropout_50(out)
        out = self.fc_2(out)
        return out    
    

class Modified_VGG6(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(Modified_VGG6, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(512*2*2, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.dropout_50(out)
        out = self.fc_2(out)
        return out
    
class FullyConvolutionalResnet18(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
 
        # Start with standard resnet18 defined here 
        super().__init__(block = models.resnet.BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url( models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)
 
        # Replace AdaptiveAvgPool2d with standard AvgPool2d 
        self.avgpool = nn.AvgPool2d((7, 7))
 
        # Convert the original fc layer to a convolutional layer.  
        self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_ (self.fc.bias.data)
 
    # Reimplementing forward pass. 
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
 
        # Notice, there is no forward pass 
        # through the original fully connected layer. 
        # Instead, we forward pass through the last conv layer
        x = self.last_conv(x)
        return x
    
    
    
class SimpleNet(nn.Module):
    def __init__(self, num_classes,input_channel=3):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.MaxPool2d(kernel_size = 2)(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.Dropout2d()(x)
        x = nn.MaxPool2d(kernel_size = 2)(x)
        x = nn.ReLU()(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        return x
    
    
# Simple ResNet
class Simple_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(Simple_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
# Modified ResNet
class Modified_ResNet(nn.Module):
    def __init__(self, block, layers, num_channels, num_classes = 10):
        super(Modified_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 128, layers[0], stride = 1)
        #self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        #self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc1 = nn.Linear(28800, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        

        return x    
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    
class Siamese_test_model(nn.Module):

    def __init__(self, input_channel=3):
        super(Siamese_test_model, self).__init__()

        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.flatten_1 = nn.Flatten()
        self.cls_head = nn.Linear(800, 1) #nn.Sequential([nn.Linear(8192, 800),nn.Linear(800, 1)]) # nn.Sequential(
  
    def backbone(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.flatten_1(out)

        return out
    
    
    def forward_one(self, x):
        x = self.backbone(x)
        #print(f'Size after backbone {x.size()}')
        x = x.view(x.size()[0], -1)
        #print(f'Size after view {x.size()}')
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = out1*out2
        #print(f'size {out.size}')
        out = self.cls_head(out)
        out = nn.Sigmoid()(out)

        return out
    
class Siamese_size64_model(nn.Module):

    def __init__(self, input_channel=3):
        super(Siamese_size64_model, self).__init__()

        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.flatten_1 = nn.Flatten()
        self.cls_head = nn.Sequential(nn.Linear(8192, 4096),nn.Sigmoid()) # nn.Sequential(
        self.pred_head = nn.Linear(4096, 1)
    def backbone(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.flatten_1(out)

        return out
    
    
    def forward_one(self, x):
        x = self.backbone(x)
        #print(f'Size after backbone {x.size()}')
        x = x.view(x.size()[0], -1)
        #print(f'Size after view {x.size()}')
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = out1*out2
        
        out = self.cls_head(out)
        out = self.pred_head(out)

        return out

    
# Contrastive training
    
class SmallResNet18_Input64_Contrastive(nn.Module):
    def __init__(self, num_classes=1, input_channel=2):
        super(SmallResNet18_Input64_Contrastive, self).__init__()
        
        self.feature_Extractor = resnet18()
        self.feature_Extractor.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_Extractor.layer3 = nn.Identity()
        self.feature_Extractor.layer4 = nn.Identity()
        self.feature_Extractor.fc = nn.Identity()
        self.fc = nn.Linear(in_features=128, out_features=1, bias=True)
        
                
    def forward_one(self, x):
        out =  self.feature_Extractor(x)
        # out = self.conv_1(x)
        # out = self.conv_2(out)
        # out = self.dropout2d_layer(out)
        # out = self.conv_3(out)
        # out = self.conv_4(out)
        # out = self.dropout2d_layer(out)
        # out = out.reshape(out.size(0), -1)
        
        return out    
    
    def forward(self, x1, x2):
        img1 = self.forward_one(x1)
        img2 = self.forward_one(x2)
        
        out = img1*img2
        out = self.fc(out)
        return out    
    
class VGG6_Contrastive(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Contrastive, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
    def forward_one(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        
        return out    
    
    def forward(self, x1, x2):
        img1 = self.forward_one(x1)
        img2 = self.forward_one(x2)
        
        out = img1*img2
        out = self.dropout_50(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out
    
class VGG6_Input64_Contrastive_Drop20(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input64_Contrastive_Drop20, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.2)
        
    def forward_one(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        
        return out    
    
    def forward(self, x1, x2):
        img1 = self.forward_one(x1)
        img2 = self.forward_one(x2)
        
        out = img1*img2
        out = self.dropout_50(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out


    
class Modified_Siamese_test_model(nn.Module):

    def __init__(self, input_channel=3):
        super(Modified_Siamese_test_model, self).__init__()

        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.out_feature = nn.Sequential(nn.Flatten(),nn.Linear(3200, 128))
        #self.cls_head = nn.Sequential()
  
    def backbone(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.out_feature(out)
        return out
    
    
    def forward_one(self, x):
        x = self.backbone(x)
        #print(f'Size after backbone {x.size()}')
        #x = self.out_feature(x)
        
        x = x.view(x.size()[0], -1)
        #print(f'Size after view {x.size()}')
        return x
    
    def cls_head(self,x):
        out = nn.Linear(128, 64)(x)
        out = nn.Linear(64, 1)(out)

        #out = self.cls_head(out)
        out = nn.Sigmoid()(out)

        return out

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        #print(out1.size())
        out = out1*out2
        
        out = self.cls_head(out)
        return out
    
class Siamese_test_withdropout_model(nn.Module):

    def __init__(self, input_channel=3):
        super(Siamese_test_withdropout_model, self).__init__()

        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        self.flatten_1 = nn.Flatten()
        self.cls_head = nn.Sequential(nn.Linear(800, 256),nn.Linear(256, 128),nn.Dropout(0.5),nn.Linear(128, 1))
          
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
    
        
    def backbone(self, x):
        out = self.conv_1(x)
        out = self.dropout2d_layer(out)
        out = self.conv_2(out)
        out = self.flatten_1(out)

        return out
    
    
    def forward_one(self, x):
        x = self.backbone(x)
        #print(f'Size after backbone {x.size()}')
        x = x.view(x.size()[0], -1)
        #print(f'Size after view {x.size()}')
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = out1*out2
        
        out = self.cls_head(out)
        out = nn.Sigmoid()(out)

        return out
    
    
class VGG6_Input64_ConcatDiff(nn.Module):
    
    def __init__(self, num_classes=1, raw_channel=2, diff_channel=1):
        super(VGG6_Input64_ConcatDiff, self).__init__()
        
        
        # Diff extractor
        self.conv_1_diff = nn.Sequential(
            nn.Conv2d(diff_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2_diff = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3_diff = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4_diff = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1_diff= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        
        
        # 2 Raw extractor
        self.conv_1_raw = nn.Sequential(
            nn.Conv2d(raw_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2_raw = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3_raw = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4_raw = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1_raw= nn.Sequential(nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU())
        
        
        # Final prediction Head
        self.fc_2 =  nn.Linear(512, num_classes)
        # nn.Sequential(
        #     nn.Linear(512, num_classes),
        #     nn.ReLU())
        # self.fc_2 = nn.Linear(256, num_classes)
        
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward_raw(self, x):
        out = self.conv_1_raw(x)
        out = self.conv_2_raw(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3_raw(out)
        out = self.conv_4_raw(out)
        out = self.dropout2d_layer(out)
        out = out.view(x.size()[0], -1)
        out = self.dropout_50(out)
        out = self.fc_1_raw(out)
        return out
    
    def forward_diff(self, x):
        out = self.conv_1_diff(x)
        out = self.conv_2_diff(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3_diff(out)
        out = self.conv_4_diff(out)
        out = self.dropout2d_layer(out)
        out = out.view(x.size()[0], -1)
        out = self.dropout_50(out)
        out = self.fc_1_diff(out)
        return out
        
    def forward(self, x1, x2):
        
        
        raw_feature = self.forward_raw(x1)
        diff_feature = self.forward_diff(x2)
        
        # Concat at fully-connected
        out = torch.cat((raw_feature, diff_feature), 1)
        out = self.fc_2(out)
        
        
        
        return out
    
class VGG6_Input64_ContrastiveFeature(nn.Module):
    def __init__(self, num_classes=1, input_channel=3):
        super(VGG6_Input64_ContrastiveFeature, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Linear(256, num_classes)
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
    def forward_one(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        
        return out    
    
    def forward(self, x1, x2):
        img1 = self.forward_one(x1)
        img2 = self.forward_one(x2)
        
        out = img1*img2
        out = self.dropout_50(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out    
    
class VGG6_Input64_Contrastive(nn.Module):
    
    def __init__(self, num_classes=1, raw_channel=2, diff_channel=1):
        super(VGG6_Input64_Contrastive, self).__init__()
        
        
        # Similariy extractor
        
        self.conv_1_contrastive = nn.Sequential(
            nn.Conv2d(diff_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2_contrastive = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3_contrastive = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4_contrastive = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        self.fc_1_contrastive = nn.Sequential(nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU())
             
        # Raw extractor
        self.conv_1_raw = nn.Sequential(
            nn.Conv2d(raw_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2_raw = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3_raw = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4_raw = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1_raw= nn.Sequential(nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU())
        
        
        # Final prediction Head
        self.fc_2 =  nn.Linear(512, num_classes)
        # nn.Sequential(
        #     nn.Linear(512, num_classes),
        #     nn.ReLU())
        # self.fc_2 = nn.Linear(256, num_classes)
        
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        
    def forward_contrastive(self, x):
        out = self.conv_1_contrastive(x)
        out = self.conv_2_contrastive(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3_contrastive(out)
        out = self.conv_4_contrastive(out)
        out = self.dropout2d_layer(out)
        return out
    
    def forward_raw(self, x):
        out = self.conv_1_raw(x)
        out = self.conv_2_raw(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3_raw(out)
        out = self.conv_4_raw(out)
        out = self.dropout2d_layer(out)
        out = self.dropout_50(out)
        out = self.fc_1_raw(out)
        return out
        
    def forward(self, x):
        
        # n2n feature
        raw_feature = self.forward_raw(x)
        
        
        # Similarity features
        contrastive_feature_1 = self.forward_contrastive(x[:,:1]) # Template images        
        contrastive_feature_2 = self.forward_contrastive(x[:,1:2]) # Science images
        
        similarity_feature = contrastive_feature_1*contrastive_feature_2
        
        similarity_feature = self.fc_1_contrastive(similarity_feature)
        # Concat at fully-connected
        out = torch.cat((raw_feature, similarity_feature), 1)
        out = self.fc_2(out)
        
        return out
    
class ResNet18_Input64_Contrastive(nn.Module):
    
    def __init__(self, num_classes=1, raw_channel=2, diff_channel=1):
        super(ResNet18_Input64_Contrastive, self).__init__()
        
        # ================== Raw extractor ================== #
        
        self.feature_Extractor_raw = resnet18()
        self.feature_Extractor_raw.conv1 = nn.Conv2d(raw_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_Extractor_raw.layer3 = nn.Identity()
        self.feature_Extractor_raw.layer4 = nn.Identity()
        self.feature_Extractor_raw.fc = nn.Identity()
        
        
        
        # Similariy extractor
        
        self.feature_Extractor_diff = resnet18()
        self.feature_Extractor_diff.conv1 = nn.Conv2d(diff_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_Extractor_diff.layer3 = nn.Identity()
        self.feature_Extractor_diff.layer4 = nn.Identity()
        self.feature_Extractor_diff.fc = nn.Identity()
        
             
       
        # Final linear
        self.fc = nn.Linear(in_features=256, out_features=1, bias=True)
        
        
    def forward_contrastive(self, x):
        out =  self.feature_Extractor_diff(x)
        return out
    
    def forward_raw(self, x):
        out =  self.feature_Extractor_raw(x)
        return out
        
    def forward(self, x):
        
        # n2n feature
        raw_feature = self.forward_raw(x)
        
        
        # Similarity features
        contrastive_feature_1 = self.forward_contrastive(x[:,:1]) # Template images        
        contrastive_feature_2 = self.forward_contrastive(x[:,1:2]) # Science images
        
        similarity_feature = contrastive_feature_1*contrastive_feature_2
        
        # similarity_feature = self.fc_1_contrastive(similarity_feature)
        
        # Concat at fully-connected
        out = torch.cat((raw_feature, similarity_feature), 1)
        out = self.fc(out)
        
        return out
    
    
#============== Contrastive + Classification model ==============#
        
class VGG6_Input64_with_pretrained_ResNet18_contrastive(nn.Module):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3, freeze=True):
        super(VGG6_Input64_with_pretrained_ResNet18_contrastive, self).__init__()
        
        self.extractor = contrastive_extractor ## Resnet
        
        
        # Classification : VGG6
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        self.fc_1= nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU())
        
        # Additional layer to output 1x128 features
        self.fc_2= nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        
        self.fc_3 = nn.Linear(128*2, num_classes) # Just added 04/09/23
        
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        self.freeze = freeze
    
        
        
    def forward(self, x):
        
        contrastive_out = self.extractor.forward_one(x)
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout_50(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        out = self.fc_3(out)
        
        
        return out   
    
class VGG6_Input64_with_pretrained_VGG6_contrastive(nn.Module):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3, freeze=True):
        super(VGG6_Input64_with_pretrained_VGG6_contrastive, self).__init__()
        
        self.extractor = contrastive_extractor # VGG6
        
        
        
        # Classification : VGG6
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4))
        
        
        # Contrastive no RElu
        # self.fc_1 = nn.Linear(2048, 128)
        # nn.Sequential(nn.Linear(2048, 128),nn.ReLU())
        
        self.fc_2 = nn.Linear(2048*2, num_classes)
        # self.fc_3 = nn.Linear(128, num_classes) # Just added 04/09/23
        
        
        self.dropout2d_layer = nn.Dropout2d(0.25)
        
        self.dropout_50 = nn.Dropout(0.5)
        
        # self.freeze = freeze
    
        
        
    def forward(self, x):
        
        contrastive_out = self.extractor.forward_one(x)
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout2d_layer(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.dropout2d_layer(out)
        out = out.reshape(out.size(0), -1)

        # print(out.size())
        # print(contrastive_out.size())
        
        out = torch.cat((out, contrastive_out), 1)
        
        out = self.fc_2(out)
        # out = self.fc_3(out)
        
        
        return out   
  

class ResNet18_Input64_with_pretrained_VGG6_contrastive(nn.Module):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3):
        super(ResNet18_Input64_with_pretrained_VGG6_contrastive, self).__init__()
        
        self.extractor = contrastive_extractor
        
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.layer3= nn.Identity()
        self.model.layer4= nn.Identity()
        # model.avgpool = nn.Identity()
        
        
        # self.fc_1 = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=num_classes, bias=True),
        #     nn.ReLU())
        
        # Additional layer to get the same shape as VGG6
        self.model.fc = nn.Linear(in_features=128, out_features=2048, bias=True) # additional layer to upsampling
        
        self.final_fc = nn.Linear(in_features=2048*2, out_features=num_classes, bias=True)
        
        
    def forward(self, x):
        
        contrastive_out = self.extractor.forward_one(x)
        out = self.model(x)
        # print(out.size())
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        
        out = self.final_fc(out)
        # out = self.fc_3(out)
        
        return out
    
    
class ResNet18_Input64_with_pretrained_ResNet18_contrastive(nn.Module):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3):
        super(ResNet18_Input64_with_pretrained_ResNet18_contrastive, self).__init__()
        
        self.extractor = contrastive_extractor
        
        self.model = resnet18()
        self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.layer3= nn.Identity()
        self.model.layer4= nn.Identity()
        # model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        
        
        # self.fc_1 = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=num_classes, bias=True),
        #     nn.ReLU())
        
        
        self.final_fc = nn.Linear(in_features=128*2, out_features=num_classes, bias=True)
        
        
    def forward(self, x):
        
        contrastive_out = self.extractor.forward_one(x)
        out = self.model(x)
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        
        out = self.final_fc(out)
        # out = self.fc_3(out)
        
        return out
        

## =========== Baseline Models =========== ##    

# OTNet_CNN : modified to our input
class Baseline_CNN_OTNet_Input80(nn.Module):
    def __init__(self, num_classes=2, input_channel=2):
        super(Baseline_CNN_OTNet_Input80, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 3, stride=3))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2))
        
        self.fc_1 = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        
        self.fc_3 = nn.Sequential(
            nn.Linear(128, num_classes),
            )
        # nn.Softmax(dim=-2)
        self.flatten = nn.Flatten()
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.50)
        self.dropout_3 = nn.Dropout(0.30)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout_1(out)
        out = self.conv_3(out)
        out = self.flatten(out)
        out = self.fc_1(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        out = self.dropout_3(out)
        out = self.fc_3(out)
        
        return out


# OTNet_CNN : modified to our input
class Baseline_CNN_OTNet(nn.Module):
    def __init__(self, num_classes=2, input_channel=2):
        super(Baseline_CNN_OTNet, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 3, stride=3))
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2))
        
        self.fc_1 = nn.Sequential(
            nn.Linear(800, 256),
            nn.ReLU())
        
        self.fc_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        
        self.fc_3 = nn.Sequential(
            nn.Linear(128, num_classes),
            )
        # nn.Softmax(dim=-2)
        self.flatten = nn.Flatten()
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.50)
        self.dropout_3 = nn.Dropout(0.30)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout_1(out)
        out = self.conv_3(out)
        out = self.flatten(out)
        out = self.fc_1(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        out = self.dropout_3(out)
        out = self.fc_3(out)
        
        return out
    
# OTNet_CNN : modified to our input
class Baseline_DLN_OTNet(nn.Module):
    def __init__(self, num_classes=1, input_channel=2):
        super(Baseline_DLN_OTNet, self).__init__()
        

        # input 1
        self.dense_input1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU())
        
        self.dense_input2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU())
        
        self.fc_1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU())
        self.fc_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU())
        self.fc_3 = nn.Sequential(
            nn.Linear(64, num_classes),
            )
        #nn.Softmax(dim=-2)
        # self.flatten = nn.Flatten()
        
        # self.softmax = nn.Softmax(dim=1)
        
        
        
        
    def forward(self, x):
        out_0 = self.dense_input1(x[:,0])
        out_1 = self.dense_input2(x[:,1])
        
        out = torch.cat((out_0, out_1), 1)
        out = nn.Flatten()(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
  
        return out


# ==== Add constrastive feature ==== #
class OTCNN_Input64_with_pretrained_ResNet18_contrastive(Baseline_CNN_OTNet):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3):
        Baseline_CNN_OTNet.__init__(self)
        
        self.extractor = contrastive_extractor
        
        self.final_fc = nn.Sequential(
            nn.Linear(256, num_classes),
            )
        
    def forward(self, x):
        
        # Contrastive feature
        contrastive_out = self.extractor.forward_one(x)
        
        # end-to-end feature
        
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout_1(out)
        out = self.conv_3(out)
        out = self.flatten(out)
        out = self.fc_1(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        out = self.dropout_3(out)
        # out = self.fc_3(out)
        
        
        
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        
        
        # out = self.fc_2(out)
        out = self.final_fc(out)
        
        return out
    
# Pretrained VGG6
class OTCNN_Input64_with_pretrained_VGG6_contrastive(Baseline_CNN_OTNet):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3):
        Baseline_CNN_OTNet.__init__(self)
        
        self.extractor = contrastive_extractor
        
        self.final_fc = nn.Sequential(
            nn.Linear(2176, num_classes),
            )
        
    def forward(self, x):
        
        # Contrastive feature
        contrastive_out = self.extractor.forward_one(x)
        
        # end-to-end feature
        
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.dropout_1(out)
        out = self.conv_3(out)
        out = self.flatten(out)
        out = self.fc_1(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        out = self.dropout_3(out)
        # out = self.fc_3(out)
        
        
        
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        
        
        # out = self.fc_2(out)
        out = self.final_fc(out)
        
        return out
    
# OTNet_CNN : modified to our input
class OTDLN_Input64_with_pretrained_ResNet18_contrastive(Baseline_DLN_OTNet):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3):
        super(OTDLN_Input64_with_pretrained_ResNet18_contrastive, self).__init__()
        Baseline_DLN_OTNet.__init__(self)
        
        self.extractor = contrastive_extractor
        
        self.final_fc = nn.Sequential(
            nn.Linear(192, num_classes),
            )
        
    def forward(self, x):
        
        # Contrastive feature
        contrastive_out = self.extractor.forward_one(x)
        
        # end-to-end feature
        
        out_0 = self.dense_input1(x[:,0])
        out_1 = self.dense_input2(x[:,1])
        
        out = torch.cat((out_0, out_1), 1)
        out = nn.Flatten()(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        # out = self.fc_3(out)
        
        
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        
        
        # out = self.fc_2(out)
        out = self.final_fc(out)
        
        return out

# Pretrained VGG6
class OTDLN_Input64_with_pretrained_VGG6_contrastive(Baseline_DLN_OTNet):
    def __init__(self, contrastive_extractor, num_classes=1, input_channel=3):
        super(OTDLN_Input64_with_pretrained_VGG6_contrastive, self).__init__()
        Baseline_DLN_OTNet.__init__(self)
        
        self.extractor = contrastive_extractor
        
        self.final_fc = nn.Sequential(
            nn.Linear( 2112, num_classes),
            )
        
    def forward(self, x):
        
        # Contrastive feature
        contrastive_out = self.extractor.forward_one(x)
        
        # end-to-end feature
        
        out_0 = self.dense_input1(x[:,0])
        out_1 = self.dense_input2(x[:,1])
        
        out = torch.cat((out_0, out_1), 1)
        out = nn.Flatten()(out)
        out = self.fc_1(out)
        out = self.fc_2(out)
        # out = self.fc_3(out)
        
        
        # out = out*contrastive_out
        out = torch.cat((out, contrastive_out), 1)
        
        
        # out = self.fc_2(out)
        out = self.final_fc(out)
        
        return out
