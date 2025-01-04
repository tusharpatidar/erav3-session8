import torch
import torch.nn as nn

class DepthwiseConv(nn.Module):
    """Custom implementation of depthwise convolution"""
    def __init__(self, in_channels, kernel_size, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            dilation=dilation
        )
    
    def forward(self, x):
        return self.conv(x)

class PointwiseConv(nn.Module):
    """Custom implementation of pointwise convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    """Custom convolution block with optional dilation and depthwise separation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 dilation=1, use_depthwise=False):
        super().__init__()
        
        if use_depthwise:
            self.conv = nn.Sequential(
                DepthwiseConv(in_channels, kernel_size, 
                             padding=dilation, dilation=dilation),
                PointwiseConv(in_channels, out_channels)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 padding=dilation, dilation=dilation)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CIFAR10Model(nn.Module):
    """
    Custom CNN for CIFAR10 using dilated convolutions
    - Maintains spatial dimensions using dilated convolutions
    - Uses depthwise separable convolutions for efficiency
    - Progressive increase in receptive field
    """
    def __init__(self):
        super().__init__()
        
        # Initial convolution block (reduced channels)
        self.initial_block = nn.Sequential(
            ConvBlock(3, 12),  # Reduced from 16
            ConvBlock(12, 16, dilation=2)  # Reduced from 24
        )
        
        # Middle block with depthwise separable conv
        self.middle_block = nn.Sequential(
            ConvBlock(16, 24, use_depthwise=True),  # Reduced from 32
            ConvBlock(24, 32, dilation=4)  # Reduced from 48
        )
        
        # Deep features block
        self.feature_block = nn.Sequential(
            ConvBlock(32, 48, dilation=8),  # Reduced from 64
            ConvBlock(48, 64, dilation=16)  # Reduced from 96
        )
        
        # Final block (reduced channels)
        self.final_block = ConvBlock(64, 96, dilation=32)  # Reduced from 128
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 10)  # Reduced from 128
        )
    
    def forward(self, x):
        x = self.initial_block(x)    # RF: 5 -> 9
        x = self.middle_block(x)     # RF: 11 -> 19
        x = self.feature_block(x)    # RF: 35 -> 67
        x = self.final_block(x)      # RF: 131
        return self.classifier(x)
