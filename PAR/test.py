import os
import torch.profiler
from torch.utils.data import DataLoader
import torchvision.transforms as T
from loss_fn.MaskedLoss import *
from net.MultitaskNet import *
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from Dataset import *
from DataTracker import DataTracker
from CustomSampler import *
import time
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import random
import numpy as np

'''
    This file performs the testing of a model on a testing dataset, the default testing dataset is out own custom dataset.
    If a new dataset is created it should respect the same labeling structure as the old dataset
'''

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_METRICS_DIR = './models_v2'                          # Output folder for saving the computed metrics
MODEL = './models/Finale3_pth'                              # Path to the model file

def visualize_random_transformations(image_folder, transform, k=5):
    """
    Visualize original and transformed random images from a folder.
    
    Args:
        image_folder: str, path to the folder containing images
        transform: torchvision transforms
        k: number of random images to display
    """
    i
    
    # Get list of image files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        f for f in os.listdir(image_folder) 
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    # Make sure k is not larger than available images
    k = min(k, len(image_files))
    
    # Randomly select k images
    random.shuffle(image_files)
    selected_images = random.sample(image_files, k)
    
    # Create a figure with 2 rows (original and transformed) and k columns
    fig, axes = plt.subplots(2, k, figsize=(4*k, 8))
    
    # If k=1, axes will be 1D, we need to reshape it to 2D
    if k == 1:
        axes = axes.reshape(2, 1)
    
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(image_folder, img_name)
        
        try:
            # Load and show original image
            original_img = Image.open(img_path).convert('RGB')
            axes[0, i].imshow(original_img)
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Original\n{img_name}')
            
            # Apply transformations and show transformed image
            transformed_img = transform(original_img)
            
            # Convert tensor to numpy for visualization
            # Denormalize the image
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            transformed_img = transformed_img * std + mean
            
            # Convert to numpy and transpose to (H,W,C)
            transformed_img = transformed_img.permute(1, 2, 0).numpy()
            # Clip values to [0,1] range
            transformed_img = np.clip(transformed_img, 0, 1)
            
            axes[1, i].imshow(transformed_img)
            axes[1, i].axis('off')
            axes[1, i].set_title('Transformed')
            
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            # In case of error, create empty subplot
            axes[0, i].text(0.5, 0.5, 'Error loading image', 
                          horizontalalignment='center',
                          verticalalignment='center')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../Dataset/testing/image", help="Testing images directory")
    parser.add_argument("--labels", type=str, default="../Dataset/testing/testing.txt", help="Testing labels directory")
    parser.add_argument("--batch_size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument("--load_model", type=str, default="./models/multitask_masked_loss_resnet50.pth", help="Path to the model to use")
    parser.add_argument("--soglia", type=float, default=0.5, help="Threshold for classification")

    args = parser.parse_args()


    print("-"*100)
    print(f"Loading {args.load_model}...")
    model = MultitaskCNNNet(train_backbone=False)
    model.load_model(args.load_model)
    model = model.to(DEVICE)
    model.eval()

    # Initialize dataset, dataloader, model
    
    transform = T.Compose([
        T.Resize((220,96)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    dataset = CustomDataset(args.image, args.labels, transform=transform)

    print("Initializing dataloaders...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=12)
    print("Done preparing the dataloader...")
    
    # Decomment to visualize the transformation applied to the images
    #visualize_random_transformations(args.image, transform, k=5)
    
    metrics = DataTracker(
                 training_metrics_file=None,
                 validation_metrics_file=f"{OUTPUT_METRICS_DIR}/testing_result.csv", 
                 tn_fp_fn_tp_file_train=None,
                 tn_fp_fn_tp_file_val=f"{OUTPUT_METRICS_DIR}/testing_tn_fp_fn_tp.csv", 
                 training_losses_file=None,
                 validation_losses_file=None,
                 verbose=True)
    inference_time = []

    total_time = 0.0
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            start = time.time()
            for image, gender_labels, bag_labels, hat_labels in dataloader:
                image, gender_labels, bag_labels, hat_labels = image.to(DEVICE), gender_labels.to(DEVICE), bag_labels.to(DEVICE), hat_labels.to(DEVICE) 
                start_inference = time.time()
                gender_logit, bag_logit, hat_logit = model(image)

                end_inference = time.time()
                inference_time.append(time.time() - start_inference)
                metrics.update_counter([gender_logit, bag_logit, hat_logit], [gender_labels, bag_labels, hat_labels], phase="validation", soglia=[args.soglia, args.soglia, args.soglia])
                pbar.update(1)            
            total_time  = time.time()-start
            metrics.compute_metrics(phase="validation")
            accuracy, precision, recall, f1Score, balanced_accuracy =  metrics.get_metrics(phase="validation")
    
            

    print("-------- Testing Results --------")
    print("Metrics:")
    print(f"\tGender :\n\t\trecall : {recall['gender'] * 100}\n\t\taccuracy : {accuracy['gender'] * 100}\n\t\tF1Score : {f1Score['gender'] * 100}\n\t\tPrecision : {precision['gender'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['gender']}")
    print(f"\n\tBag : \n\t\trecall : {recall['bag'] * 100}\n\t\taccuracy : {accuracy['bag'] * 100}\n\t\tF1Score : {f1Score['bag'] * 100}\n\t\tPrecision : {precision['bag'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['bag']}")
    print(f"\n\tHat : \n\t\trecall : {recall['hat'] * 100}\n\t\taccuracy : {accuracy['hat'] * 100}\n\t\tF1Score : {f1Score['hat'] * 100}\n\t\tPrecision : {precision['hat'] * 100}\n\t\tbalanced_accuracy : {balanced_accuracy['hat']}")
    print("\n\n--------- Testing Time ---------")
    print(f"Total time : {total_time} seconds")
    print(f"Mean Inference time : {sum(inference_time) / len(inference_time)} seconds")

    print("-------------------------------END TEST --------------------------------")
    print("\n\n\n")