"""
data_loader.py - Dataset loading, preprocessing, and augmentation for SOD.

Handles:
  - Loading image/mask pairs from disk
  - Resizing to 128x128 and normalizing to [0, 1]
  - 70/15/15 train/val/test split
  - Augmentations for training: horizontal flip + brightness/contrast jitter
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SaliencyDataset(Dataset):
    """
    PyTorch Dataset for salient object detection.

    Loads paired RGB images and binary grayscale masks.
    Applies separate transforms to images; masks are always resized + converted to tensor.
    """

    def __init__(self, img_dir, mask_dir, filenames, transform=None):
        """
        Args:
            img_dir   (str):  Path to folder containing input images.
            mask_dir  (str):  Path to folder containing ground-truth masks.
            filenames (list): List of image filenames to include in this split.
            transform:        torchvision transform pipeline applied to images only.
        """
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.filenames = filenames
        self.transform = transform

        # Mask transform: always resize + convert to tensor (no augmentation)
        self.mask_transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name  = self.filenames[idx]
        img_path  = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        # Support masks saved as either .png or .jpg
        mask_path = os.path.join(self.mask_dir, base_name + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base_name + ".jpg")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")   # grayscale

        if self.transform:
            image = self.transform(image)

        mask = self.mask_transform(mask)

        return image, mask


def load_data(batch_size=32):
    """
    Build train, validation, and test DataLoaders from the dataset folder.

    Split: 70% train / 15% validation / 15% test.
    Training images receive augmentation; val/test images are only resized.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    img_folder  = "dataset/images"
    mask_folder = "dataset/masks"

    files = sorted(os.listdir(img_folder))

    # 70 / 15 / 15 split
    train_end = int(0.70 * len(files))
    val_end   = int(0.85 * len(files))

    train_files = files[:train_end]
    val_files   = files[train_end:val_end]
    test_files  = files[val_end:]

    print(f"Dataset split — Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Training augmentations: flip + brightness/contrast jitter
    train_transform = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3),   # brightness variation
        T.ToTensor(),
    ])

    # Validation/test: only resize — no augmentation
    eval_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    train_loader = DataLoader(
        SaliencyDataset(img_folder, mask_folder, train_files, train_transform),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SaliencyDataset(img_folder, mask_folder, val_files, eval_transform),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        SaliencyDataset(img_folder, mask_folder, test_files, eval_transform),
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader