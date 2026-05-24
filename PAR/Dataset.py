from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os

class CustomDataset(Dataset):

    def __init__(self, image_folder, labels_file, transform=None, max_samples=-1):
        '''
            Labels file structure:
                image name, gender label, bag label, hat label
        '''
        self.images_dir = image_folder
        self.transform = transform
        self.counter = {
            'gender' : {0:0, 1:0, -1:0},
            'bag' : {0:0, 1:0, -1:0},
            'hat' : {0:0, 1:0, -1:0}
        }

        self.data = []
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                image_name = parts[0]
                labels = list(map(int, parts[3:]))  # Converte le etichette in interi
                if os.path.exists(os.path.join(self.images_dir, image_name)):
                    self.data.append((image_name, labels))
                    self.counter['gender'][labels[0]] += 1
                    self.counter['bag'][labels[1]] += 1
                    self.counter['hat'][labels[2]] += 1
        
        if max_samples > 0:
            np.random.shuffle(self.data)
            self.data = self.data[:max_samples]

    
    def __getitem__(self, idx):
        image_name, labels = self.data[idx]
        image_path = os.path.normcase(os.path.join(self.images_dir, image_name))
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        gender = torch.tensor(labels[0])
        bag = torch.tensor(labels[1])
        hat = torch.tensor(labels[2])

        return image, gender, bag, hat

    def _calculate_relative_frequencies(self):
        """
        Calculate the relative frequencies of each class for each task in the dataset.

        The relative frequency of a class is defined as the ratio of the number of samples in that class to the total number of samples in the dataset.
        Missing labels (with value -1) are excluded from the calculation.

        Returns:
            list: A list of torch tensors, where each tensor represents the relative frequencies for a specific task.
            The length of the list is equal to the number of tasks.
        """
        num_tasks = 3
        # Initialize a list to store the relative frequencies for each task
        relative_frequencies = [[] for _ in range(num_tasks)]

        # Iterate over all samples in the dataset
        for _, labels in self.data:
            for task_idx in range(num_tasks):
                task_label = labels[task_idx]  # Label for the current task

                # Exclude missing labels (value -1)
                if task_label != -1:
                    relative_frequencies[task_idx].append(task_label)

        # Calculate the relative frequency for each task
        for task_idx in range(num_tasks):
            task_labels = torch.tensor(relative_frequencies[task_idx])  # Labels for the current task
            class_counts = torch.bincount(task_labels)  # Count the absolute frequency of each class
            task_relative_frequencies = class_counts.float() / len(task_labels)  # Relative frequency
            relative_frequencies[task_idx] = task_relative_frequencies

        return relative_frequencies


    def calculate_class_weights(self):
        """
        Calculate class weights for each task in the dataset based on the relative frequencies of each class.

        This method first calls the `_calculate_relative_frequencies` method to obtain the relative frequencies of each class for each task.
        Then, it calculates the inverse of the relative frequencies to obtain the class weights.
        Optionally, it normalizes the weights by dividing each weight by the sum of all weights for that task.

        Returns:
            list: A list of torch tensors, where each tensor represents the class weights for a specific task.
            The length of the list is equal to the number of tasks.
        """
        relative_frequencies = self._calculate_relative_frequencies()
        class_weights = []

        for task_idx in range(len(relative_frequencies)):
            # Calculate the inverse of the relative frequencies
            task_weights = 1.0 / relative_frequencies[task_idx]

            # Optional: Normalize the weights
            task_weights = task_weights / task_weights.sum()

            class_weights.append(task_weights)

        return class_weights

    def __len__(self):
        return len(self.data)