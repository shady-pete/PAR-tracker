import torch
import torch.nn as nn
from .Backbone import *
from .ClassificationHead import *


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

class MultiTaskNet(nn.Module):
    def __init__(self, train_backbone=True):
        super(MultiTaskNet, self).__init__()
        self.backbone = ResNext50_32x4d(trainable=train_backbone)
        self.backbone_output_shape = self.backbone.get_output_shape()
        self.shared_features = self.backbone.features

        self.gender_head = BinaryClassificationHead(self.backbone_output_shape)
        self.bag_head = BinaryClassificationHead(self.backbone_output_shape, 0.1)
        self.hat_head = BinaryClassificationHead(self.backbone_output_shape, 0.1)       


    def forward(self, x):
        shared_features = self.backbone(x)

        gender_logits = self.gender_head(shared_features)
        bag_logits = self.bag_head(shared_features)
        hat_logits = self.hat_head(shared_features)

        return gender_logits, bag_logits, hat_logits

    def save_model(self, file_path):
        """
        Save the model state_dict to the specified path.
        """
        # Salva il dizionario nel file
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Load the model state_dict from the specified path.
        """
        self.load_state_dict(torch.load(file_path, weights_only=True))

    def get_trasformation(self):
        return self.backbone.get_required_tranform()
    
    def trian(self):
        self.backbone.train()
        self.gender_head.train()
        self.bag_head.train()
        self.hat_head.train()
    
    def val(self):
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