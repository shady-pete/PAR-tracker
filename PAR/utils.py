import argparse
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torchvision.transforms as T
from matplotlib.patches import Patch

from Dataset import CustomDataset  

def calculate_average_image_size(dataset):
    """
    Calculate the average width and height of images in the dataset.

    Args:
        dataset (CustomDataset): The dataset containing the images.

    Returns:
        tuple: The average width and height.
    """
    total_width = 0
    total_height = 0
    total_images = len(dataset)

    for image, _, _, _ in dataset:
        _, height, width = image.shape  # Tensor shape (C, H, W)

        total_width += width
        total_height += height

    avg_width = total_width / total_images
    avg_height = total_height / total_images

    return avg_width, avg_height


# Funzione per generare istogrammi
def plot_histogram(data, task_name, labels):
    """
    Genera un istogramma per un determinato task.

    Args:
        data (list): Lista di valori corrispondenti al task.
        task_name (str): Nome del task.
        labels (list): Etichette per le categorie del task.
    """

    count = [0, 0, 0]
    for i in data:
        if str(i) == labels[0]:
            count[0] += 1
        if str(i) == labels[1]:
            count[1] += 1
        if str(i) == labels[2]:
            count[2] += 1
    
    bar_colors = ['tab:blue', 'tab:orange','tab:red']
    fig, ax = plt.subplots()

    bars = ax.bar(labels, count, color = bar_colors, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, str(int(height)), ha='center', va='bottom', fontweight='bold')
    
    if task_name == "Gender":
        labels = ["Male", "Female", "Missing label"]
    else:
        labels = ["Not Present", "Present", "Missing label"]

    handles = [Patch(color='tab:blue', label=labels[0]),
           Patch(color='tab:orange', label=labels[1]),
           Patch(color='tab:red', label=labels[2])]

    ax.legend(handles=handles)


    ax.set_title(f"Istogramma - {task_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel(task_name, fontsize=12)
    ax.set_ylabel("Frequenza", fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../Dataset/testing/image", help="Images directory")
    parser.add_argument("--labels", type=str, default="../Dataset/testing/testing.txt", help="txt file containing labels")

    args = parser.parse_args()


    print("Loading datasets...")
    transform = T.Compose([
        T.ToTensor()                # Only transformation needed
    ])

    dataset = CustomDataset(args.image, args.labels, transform=transform)


    print("Initializing dataloaders...")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    avg_width, avg_height = calculate_average_image_size(dataset)
    print(f"Average image size - Width: {avg_width:.2f} px, Height: {avg_height:.2f} px")


    # Histogram generation

    genere_data = []
    bag_presence_data = []
    hat_data = []

    for data in dataloader:
        image, gender, bag, hat = data 
        genere_data.append(gender.item()) 
        bag_presence_data.append(bag.item())
        hat_data.append(hat.item())


    genere_labels = ["0", "1", "-1"]
    bag_presence_labels = ["0", "1", "-1"]
    hat_labels = ["0", "1", "-1"]

    plot_histogram(genere_data, "Gender", genere_labels)
    plot_histogram(bag_presence_data, "Bag Presence", bag_presence_labels)
    plot_histogram(hat_data, "Hat", hat_labels)

