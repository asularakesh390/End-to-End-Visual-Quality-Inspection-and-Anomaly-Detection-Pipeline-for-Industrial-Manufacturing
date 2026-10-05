import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    """
    Extracts mid-level and high-level feature representations 
    from a pre-trained ResNet backbone for anomaly scoring.
    """
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        
        # Extract features from intermediate layers
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Low-level features
        self.layer2 = resnet.layer2                                # Mid-level features
        self.layer3 = resnet.layer3                                # High-level features
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]
