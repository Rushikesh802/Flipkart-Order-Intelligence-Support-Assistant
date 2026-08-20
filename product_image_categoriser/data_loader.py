import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'fashion_mnist')

def get_dataloaders(data_dir=DEFAULT_DATA_DIR, batch_size=64, val_size=5000, random_seed=42):
    """
    Downloads FashionMNIST, creates a stratified train/val split, and returns DataLoaders.
    The validation set will have `val_size` images (default 5000) stratified across classes.
    """
    # Define transformations for a pretrained ImageNet backbone (e.g. ResNet)
    # Expected input size is 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download/load the training data
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    # Download/load the test data
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Get targets for stratified split
    targets = full_train_dataset.targets
    
    # Create train and validation indices using a stratified split
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_size,
        random_state=random_seed,
        stratify=targets
    )

    # Create Subset objects
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_train_dataset.classes

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    
    print("Loading Fashion-MNIST dataset...")
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    print("\nDataset successfully loaded and split!")
    print(f"Classes ({len(classes)}): {classes}")
    
    # Calculate dataset sizes
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    print(f"\nSplit sizes:")
    print(f"Training set:   {train_size:,} images")
    print(f"Validation set: {val_size:,} images")
    print(f"Test set:       {test_size:,} images")
    print(f"Total:          {train_size + val_size + test_size:,} images")
