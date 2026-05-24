from torchsummary import summary
from torchvision import models
from net.MultitaskNet import *
import net.Backbone as Backbone
from net.ClassificationHead import CNNClassificationHead

def print_network_structures():
    """
    This function prints the structure of the backbone, classification head, and the entire network.
    It uses the torchsummary library to display the summary of each component.

    Parameters:
    None

    Returns:
    None
    """
    space = '='*50
    print(f"{space} Backbone structure {space}")
    backbone = Backbone.Convnext().cuda()
    summary(backbone, input_size=(3, 220,96))

    print(f"{space} Classification Head structure {space}")
    classificationHead =  CNNClassificationHead().cuda()
    summary(classificationHead, input_size=(512, 13, 6))

    print(f"{space} Network structure {space}")
    model = MultitaskCNNNet().cuda()
    summary(model, input_size=(3, 220,96))
