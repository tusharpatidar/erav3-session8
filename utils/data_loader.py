import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CustomCIFAR10Dataset(Dataset):
    """Custom CIFAR10 dataset with Albumentations augmentations"""
    def __init__(self, is_train=True):
        self.dataset = datasets.CIFAR10(
            root='./data',
            train=is_train,
            download=True
        )
        
        # Calculate dataset statistics
        self.stats = {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616)
        }
        
        # Define transformations
        self.transform = self._get_transforms(is_train)
    
    def _get_transforms(self, is_train):
        if is_train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=self.stats['mean'],
                    p=0.5
                ),
                A.Normalize(
                    mean=self.stats['mean'],
                    std=self.stats['std']
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(
                    mean=self.stats['mean'],
                    std=self.stats['std']
                ),
                ToTensorV2()
            ])
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image, label
    
    def __len__(self):
        return len(self.dataset)

def get_dataloaders(batch_size=128):
    """Create and return train and test dataloaders"""
    train_dataset = CustomCIFAR10Dataset(is_train=True)
    test_dataset = CustomCIFAR10Dataset(is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader 