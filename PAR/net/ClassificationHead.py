from torch import nn
from .CBAM import CBAM

#####################################################################
##                       First attempt                             ##
#####################################################################


class ClassificationHead(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ClassificationHead, self).__init__()
        self.cbam = CBAM(channels=input_shape[0], r=2)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(input_shape[0], 1024) # Input size is now the number of channels after GAP
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512) #Consistent with input dimension
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cbam(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten after GAP
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


#####################################################################
##                         Second attempt                          ##
#####################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        return self.conv(x) + self.skip(x)


class CNNClassificationHead(nn.Module):
    def __init__(self, input_channels, num_classes, dropout=None):
        super().__init__()
        
        # CBAM attention module
        self.cbam = CBAM(128, r=2)
        
        # CNN feature extraction blocks
        self.feature_extraction = nn.Sequential(
            ConvBlock(input_channels[0], 1024),
            nn.MaxPool2d(2),
            ConvBlock(1024, 512),
            CBAM(512, r=4),
            ConvBlock(512, 256),
            CBAM(256, r=4),
            ConvBlock(256, 128)
        )
        
        # Classification layers
        if dropout is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, num_classes)
            )
            
    def forward(self, x):
        # Apply attention
        
        # Extract features
        x = self.feature_extraction(x)
        x = self.cbam(x)
        # Classify
        x = self.classifier(x)
        return x