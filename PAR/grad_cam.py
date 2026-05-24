import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
from net.ClassificationHead import ClassificationHead
import torch.nn as nn

from net.MultitaskNet import *

'''
    This code implements the Grad cam algorithm used to visualize where the attention moduls focus
'''


class GradCAMForHead:
    def __init__(self, model, target_cbam):
        self.model = model
        self.target_cbam = target_cbam
        self.gradients = None
        self.activations = None

        # Registriamo hook per catturare attivazioni e gradienti
        self.target_cbam.register_forward_hook(self._forward_hook)
        self.target_cbam.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output  # Salva le attivazioni del CBAM

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # Salva i gradienti rispetto alle attivazioni

    def forward(self, x, head):
        gender, bag, hat = self.model(x)  # Forward pass attraverso il modello
        if head == "gender":
            return F.softmax(gender, dim=1)
        if head == "hat":
            return F.softmax(hat, dim=1)

        return F.softmax(bag, dim=1)  # Restituisce le predizioni della head desiderata

    def generate_heatmap(self, target_class):
        # Calcola i pesi dei gradienti
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU per rimuovere valori negativi
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().cpu().detach().numpy()


def show_image(image_path, model, transform):
    # Seleziona il CBAM della head1
    bag_grad = GradCAMForHead(model, model.bag_head.cbam)
    gender_grad = GradCAMForHead(model, model.gender_head.cbam)
    hat_grad = GradCAMForHead(model, model.hat_head.cbam)

    # Carica e trasforma l'immagine
    image = Image.open(image_path).convert('RGB')

    input_tensor = transform(image).unsqueeze(0)  # Aggiungi batch dimension

    # Forward pass
    output_gender = gender_grad.forward(input_tensor, "gender")
    output_bag = bag_grad.forward(input_tensor, "bag")
    output_hat = hat_grad.forward(input_tensor, "hat")


    softmax = torch.nn.Softmax(dim=1)
    output_gender = softmax(output_gender)
    output_hat = softmax(output_hat)
    output_bag = softmax(output_bag)

    # Seleziona la classe target
    target_class_g = torch.argmax(output_gender).item()
    target_class_b = torch.argmax(output_bag).item()
    target_class_h = torch.argmax(output_hat).item()
    model.zero_grad()

    # Calcola il gradiente rispetto alla classe target
    output_gender[0, target_class_g].backward()
    output_bag[0, target_class_b].backward()
    output_hat[0, target_class_h].backward()

    # Genera la heatmap
    heatmap_g = gender_grad.generate_heatmap(target_class_g)
    heatmap_b = bag_grad.generate_heatmap(target_class_b)
    heatmap_h = hat_grad.generate_heatmap(target_class_h)

    # Ridimensiona e sovrapponi la heatmap
    heatmap_g = np.uint8(255 * heatmap_g)
    heatmap_g = Image.fromarray(heatmap_g).resize(image.size, Image.Resampling.LANCZOS)
    heatmap_g = np.array(heatmap_g)

    heatmap_b = np.uint8(255 * heatmap_b)
    heatmap_b = Image.fromarray(heatmap_b).resize(image.size, Image.Resampling.LANCZOS)
    heatmap_b = np.array(heatmap_b)

    heatmap_h = np.uint8(255 * heatmap_h)
    heatmap_h = Image.fromarray(heatmap_h).resize(image.size, Image.Resampling.LANCZOS)
    heatmap_h = np.array(heatmap_h)

    fig, axs = plt.subplots(1, 4, figsize=(8, 8))

    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(image)
    axs[1].imshow(heatmap_g, cmap='jet', alpha=0.5)
    axs[1].set_title(f'Head gender : {target_class_g }')
    axs[1].axis('off')

    axs[2].imshow(image)
    axs[2].imshow(heatmap_b, cmap='jet', alpha=0.5)
    axs[2].set_title(f'Head bag : {target_class_b}')
    axs[2].axis('off')

    axs[3].imshow(image)
    axs[3].imshow(heatmap_h, cmap='jet', alpha=0.5)
    axs[3].set_title(f'Head hat : {target_class_h}')
    axs[3].axis('off')

    fig.suptitle(
        str(transform),
        fontsize=8)

    os.makedirs('./images/grad_cam', exist_ok=True)
    plt.savefig(f'./images/grad_cam/{image_path.split("/")[-1]}', dpi=900)
    plt.close()


# Carica la rete
model = MultitaskCNNNet()
model.load_model("./models/Finale3.pth")
print("Done loading the model")
model.eval()
folder = "../../Images/val"

images_from_folder = os.listdir(folder)

# folder = "../Dataset/validation/image"
for title in images_from_folder:
    image_path = f"{folder}/{title}"
    transform = torchvision.transforms.Compose([
        transforms.Resize((220, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    show_image(image_path, model, transform)

