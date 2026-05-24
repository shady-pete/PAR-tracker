import torch
import torch.nn as nn
from .Backbone import *
from .ClassificationHead import *


#####################################################################
##                       First attempt                             ##
#####################################################################

class MultiTaskNet(nn.Module):
    def __init__(self, train_backbone=True):
        super(MultiTaskNet, self).__init__()
        self.backbone = ResNet50(trainable=train_backbone)
        self.backbone_output_shape = self.backbone.get_output_shape()

        self.gender_head = ClassificationHead(self.backbone_output_shape, 2)
        self.bag_head = ClassificationHead(self.backbone_output_shape, 2)
        self.hat_head = ClassificationHead(self.backbone_output_shape,2)
        
        self.shared_features = self.backbone.features
        
    def forward(self, x):
        shared_features = self.backbone(x)
        
        gender_logits = self.gender_head(shared_features)
        bag_logits = self.bag_head(shared_features)
        hat_logits = self.hat_head(shared_features)
        
        return gender_logits, bag_logits, hat_logits
    
    def get_transformation(self):
        return self.backbone.get_required_tranform()
    
    def train(self):
        self.backbone.train()
        self.gender_head.train()
        self.bag_head.train()
        self.hat_head.train()
    
    def eval(self):
        self.backbone.eval()
        self.gender_head.eval()
        self.bag_head.eval()
        self.hat_head.eval()
    
    def get_last_shared_layers(self):
        return self.backbone.get_last_shared()
    
    def get_backbone_parameters(self):
        return self.backbone.get_shared_parameters()
    
    def get_gender_parameters(self):
        return self.gender_head.parameters()
    
    def get_bag_parameters(self):
        return self.bag_head.parameters()
    
    def get_hat_parameters(self):
        return self.hat_head.parameters()
        
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        
    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, weights_only=True))



#####################################################################
##                      Second attempt                             ##
#####################################################################

class MultitaskCNNNet(nn.Module):
    def __init__(self, train_backbone=True, dropout=[0.3, 0.3, 0.3]):
        super().__init__()
        self.backbone = Convnext(trainable=train_backbone)
        self.backbone_output_shape = self.backbone.get_output_shape()
        
        # Specific heads for each task
        self.gender_head = CNNClassificationHead(self.backbone_output_shape, num_classes=2, dropout=dropout[0])
        self.bag_head = CNNClassificationHead(self.backbone_output_shape, num_classes=2, dropout=dropout[1])
        self.hat_head = CNNClassificationHead(self.backbone_output_shape, num_classes=2, dropout=dropout[2])
        
    def forward(self, x):
        shared_features = self.backbone(x)
        
        gender_logits = self.gender_head(shared_features)
        bag_logits = self.bag_head(shared_features)
        hat_logits = self.hat_head(shared_features)
        
        return gender_logits, bag_logits, hat_logits
    
    def get_transformation(self):
        return self.backbone.get_required_tranform()
    
    def train(self):
        self.backbone.train()
        self.gender_head.train()
        self.bag_head.train()
        self.hat_head.train()
    
    def eval(self):
        self.backbone.eval()
        self.gender_head.eval()
        self.bag_head.eval()
        self.hat_head.eval()
        
    def get_backbone_parameters(self):
        return self.backbone.get_shared_parameters()
    
    def get_gender_parameters(self):
        return self.gender_head.parameters()
    
    def get_bag_parameters(self):
        return self.bag_head.parameters()
    
    def get_hat_parameters(self):
        return self.hat_head.parameters()
        
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        
    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path, weights_only=True))

