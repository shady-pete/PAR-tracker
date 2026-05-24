from net.MultitaskNet import MultitaskCNNNet
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T

class parmodel():
    def __init__(self, path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Par-model is using {self.device}")
        self.model = MultitaskCNNNet(train_backbone=False).to(self.device)
        self.model.load_model(path)
        self.model.eval()

        # transformation expected by the backbone (convnext)
        self.transform = T.Compose([
            T.Resize((220,96)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # first dummy classification to build computation graph
        starting = np.zeros((200,100,3),dtype=np.uint8)
        self.classify([starting])

    

    def classify(self, patches):  
        """
        Classifies a list of images (patches) rappresented as numpy arrays
        Returns three list of tuples, one for each classifcation task, each tuple has this format ('classification', 'probability')
        where classicification is 1 if positive and 0 if negative, and probabilty is the associated probability value (always greater than 0.5)
        """
        transformed_patches = [self.transform(Image.fromarray(img)) for img in patches]
        batch = torch.stack(transformed_patches).cuda() 

        with torch.no_grad():  
            gender_logits, bag_logits, hat_logits = self.model(batch)

        softmax = torch.nn.Softmax(dim=1)  
            
        gender_probs = softmax(gender_logits) 
        bag_probs = softmax(bag_logits)        
        hat_probs = softmax(hat_logits)
              

        gender_results = [( int(torch.argmax(gender_probs[i]).item()),          # get the index with maximum value
                            torch.max(gender_probs[i], dim=0).values.item())    # get the item with maximum value
                            for i in range(gender_probs.size(0))]               # do this for each sample in the output
        
        bag_results = [(int(torch.argmax(bag_probs[i]).item()), torch.max(bag_probs[i], dim=0).values.item()) for i in range(bag_probs.size(0))]
        hat_results = [(int(torch.argmax(hat_probs[i]).item()), torch.max(hat_probs[i], dim=0).values.item()) for i in range(hat_probs.size(0))]



        return gender_results, bag_results, hat_results