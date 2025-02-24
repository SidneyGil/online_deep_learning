import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]  # Class categories


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Pairs of images and labels (int) for classification
        You won't need to modify this, but all PyTorch datasets must implement these methods
        """
        to_tensor = transforms.ToTensor()
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:  # Check if label exists
                    image = Image.open(Path(dataset_path, fname))  # Open image
                    label_id = LABEL_NAMES.index(label)  # Convert label to index
                    self.data.append((to_tensor(image), label_id))  # Store data pair

    def __len__(self):
        return len(self.data)  # Dataset size

    def __getitem__(self, idx):
        return self.data[idx]  # Retrieve a sample


def load_data(dataset_path: str, num_workers: int = 0, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True)  # Load data


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)  # Get predicted class
    return (outputs_idx == labels).float().mean()  # Compute accuracy
