from torch import nn
from torchvision import models

class Convnext(nn.Module):
    output_shape = (512, 13, 6)
    def __init__(self, weights=models.ConvNeXt_Base_Weights.DEFAULT, trainable=True):
        super(Convnext,self).__init__()
        self.convnext = models.convnext_base(weights=weights)

        original_features = list(self.convnext.children())[0]
        
        # ConvNext has 4 stages, we want to keep first 3 stages (3/4 of the network)
        # Each stage is a Sequential containing downsample and blocks
        stages = list(original_features)
        kept_stages = stages[:6]  # Keep first 3 stages (each stage has 2 components)
        
        # Create new sequential with only the stages we want to keep
        self.features = nn.Sequential(*kept_stages)

#        self.features = nn.Sequential(*list(self.convnext.children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.features[-1][-1].block[0].parameters():
            param.requires_grad = True

    def get_shared_parameters(self):
        return self.features.parameters()

    def forward(self, x):
        return self.features(x)

    def get_output_shape(self):
        return self.output_shape

    def get_required_tranform(self):
        return models.ConvNeXt_Base_Weights.DEFAULT.transforms()
    
    def get_last_shared(self):
        return [self.features[-1][-1].block[0]]

class ResNet50(nn.Module):
    output_shape = (2048, 7, 7)  # Ottenuta guardando all'architettura della rete
    def __init__(self, weights=models.ResNet50_Weights.IMAGENET1K_V2, trainable=True):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=weights)
        # Elimino i layer lineari e gli AdaptiveAvgPoll2d Layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        for param in self.features.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.features(x)
    def get_shared_parameters(self):
        return self.features.parameters()
    
    def get_output_shape(self):
        return self.output_shape
    
    def get_last_shared(self):
        return [self.features[-1][-1].conv3, self.features[-1][-1].conv2, self.features[-1][-1].conv1]
    
    def get_required_tranform(self):
        return models.ResNet50_Weights.IMAGENET1K_V2.transforms()