from torch import nn
from .CBAM import CBAM


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



class BinaryClassificationHead(nn.Module):

    def __init__(self, input_channels, dropout=None):
        super(BinaryClassificationHead, self).__init__()
        self.cbam = CBAM(input_channels[0], 1)
        if dropout is not None:
            print("Using dropout layer")
            self.seq = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(input_channels[0], 1024),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        else:
            self.seq = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(input_channels[0], 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

    def forward(self, x):
        x = self.cbam(x)
        x = self.seq(x)
        return x

    



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        # Proiezione 1x1 per adattare i canali nella skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        return self.block(x) + self.skip(x)

class BinaryClassificationHead2(nn.Module):

    def __init__(self, input_channels, dropout=None):
        super(BinaryClassificationHead2, self).__init__()
        self.cbam = CBAM(input_channels[0], 1)
        if dropout is not None:
            self.seq = nn.Sequential(
                nn.AdaptiveMaxPool2d((4,4)),
                nn.Flatten(),
                nn.Linear(input_channels[0]*4*4, 512),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
        else:
            self.seq = nn.Sequential(
                nn.AdaptiveAvgPool2d((2,2)),
                nn.Flatten(),
                nn.Linear(input_channels[0]*2*2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

    def forward(self, x):
        x = self.cbam(x)
        x = self.seq(x)
        return x
    
class BinaryClassificationHead3(nn.Module):

    def __init__(self, input_channels, dropout=None):
        super(BinaryClassificationHead3, self).__init__()
        self.skip1 = ResidualBlock(input_channels[0], 1024)
        self.skip2 = ResidualBlock(1024, 1024)
        self.cbam = CBAM(1024, 1)
        if dropout is not None:
            self.seq = nn.Sequential(
                nn.AdaptiveAvgPool2d((3,3)),
                nn.Flatten(),
                nn.Linear(1024*3*3, 512),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
        else:
            self.seq = nn.Sequential(
                nn.AdaptiveAvgPool2d((3,3)),
                nn.Flatten(),
                nn.Linear(1024*3*3, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

    def forward(self, x):
        x = self.skip1(x)
        x = self.skip2(x)
        x = self.cbam(x)
        x = self.seq(x)
        return x


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

class ClassificationHead2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ClassificationHead2, self).__init__()
        self.cbam = CBAM(channels=input_shape[0], r=2)
        self.flatten = nn.Flatten()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(input_shape[0], 512)  # Input size is now the number of channels after GAP
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cbam(x)
        x = self.gap(x)
        x = self.flatten(x)  # Flatten after GAP
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        return x


class HatClassification(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(HatClassification, self).__init__()
        self.cbam = CBAM(channels=input_shape[0], r=2)
        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cbam(x)
        return self.classification(x)

class BagClassification(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(BagClassification, self).__init__()
        self.cbam = CBAM(channels=input_shape[0], r=2)
        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512,  num_classes)
        )

    def forward(self, x):
        x = self.cbam(x)
        return self.classification(x)


class GenderClassification(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(GenderClassification, self).__init__()
        self.cbam = CBAM(channels=input_shape[0], r=2)
        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_shape[0], 1024),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cbam(x)
        return self.classification(x)