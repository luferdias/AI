"""
DeepLabV3+ architecture for crack segmentation.
Uses ResNet or MobileNet backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model for semantic segmentation.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        pretrained: Whether to use pretrained backbone
    """
    
    def __init__(
        self, 
        n_channels: int = 3, 
        n_classes: int = 1,
        pretrained: bool = True
    ):
        super(DeepLabV3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load pretrained DeepLabV3
        if pretrained:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=weights)
        else:
            self.model = deeplabv3_resnet50(weights=None)
        
        # Modify classifier for our number of classes
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)
        
        # Handle different input channels if needed
        if n_channels != 3:
            self.model.backbone.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.model(x)
        x = features['out']
        
        # Upsample to input size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
    
    def predict(self, x):
        """Run inference with sigmoid activation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.n_classes == 1:
                return torch.sigmoid(logits)
            else:
                return torch.softmax(logits, dim=1)


class ASPPConv(nn.Sequential):
    """Atrous Spatial Pyramid Pooling convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPP pooling module."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels: int, atrous_rates: list, out_channels: int = 256):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
