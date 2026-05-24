import torch
from torch.utils.data import Sampler
import numpy as np
from tqdm import tqdm
from threading import Thread

class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Custom Sampler ensuring each batch contains:
        - At least one sample with labels[0] != -1,
        - At least one sample with labels[1] != -1,
        - At least one sample with labels[2] != -1.

        Args:
            dataset: The dataset to sample from.
            batch_size: The batch size for each iteration.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Extract labels from the dataset
        labels = np.array([labels for _, labels in dataset.data])
        print(f"{labels[1:4,:]}")

        # Initialize arrays for valid indices using threads
        self.valid_samples = {}
        threads = []

        def calculate_valid_samples(i):
            self.valid_samples[i] = np.where(labels[:, i] != -1)[0]

        for i in range(3):
            thread = Thread(target=calculate_valid_samples, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.invalid_samples = np.where((labels == -1).all(axis=1))[0]

        # Ensure there are enough valid samples
        for i in range(3):
            if len(self.valid_samples[i]) == 0:
                raise ValueError(f"Not enough valid samples for task {i}!")

    def __iter__(self):
        """
            Iterator to yield indices for each batch while ensuring constraints.
        """
        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        all_batches = []

        # Create batches with progress tracking
        for _ in tqdm(range(dataset_size // self.batch_size), desc="Creating batches", leave=False):
            batch = []

            # Ensure at least one valid sample for each condition
            for i in range(3):
                batch.append(np.random.choice(self.valid_samples[i]))

            # Fill the rest of the batch randomly
            remaining = np.setdiff1d(indices, batch, assume_unique=True)
            batch.extend(np.random.choice(remaining, self.batch_size - len(batch), replace=False))

            all_batches.append(batch)

        # Handle remaining samples (if any)
        remaining_samples = dataset_size % self.batch_size
        if remaining_samples > 0:
            used_indices = np.concatenate(all_batches)
            remaining = np.setdiff1d(indices, used_indices, assume_unique=True)
            all_batches.append(remaining[:remaining_samples])

        # Flatten the list of batches for iteration
        flattened_batches = np.concatenate(all_batches)
        return iter(flattened_batches.tolist())

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.dataset)
