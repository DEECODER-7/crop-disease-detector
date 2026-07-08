"""
Builds PyTorch DataLoaders from the train/valid folders created in Stage 1.

We use torchvision's ImageFolder, which automatically infers class labels from
subfolder names (e.g. data/train/Apple___Black_rot/ -> label "Apple___Black_rot").
This is why the folder structure from Stage 1 matters — ImageFolder depends on it.
"""

import json
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_WORKERS,
    TRAIN_DIR,
    VALID_DIR,
)


def build_transforms(train: bool) -> transforms.Compose:
    """
    Training transforms include light augmentation (flips, rotation) so the
    model doesn't just memorize exact leaf orientations — this matters because
    real farmer-submitted photos will never look exactly like the dataset.
    Validation transforms have NO augmentation: we want to evaluate on
    realistic, unmodified images to get a true read on performance.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def get_dataloaders(subset_size: Optional[int] = None):
    """
    Returns (train_loader, valid_loader, class_names).

    subset_size: if provided, restricts BOTH train and valid sets to this many
    images (randomly). This exists purely so you can smoke-test the whole
    training pipeline in under a minute on a laptop CPU before committing to
    a real multi-hour run. Never used for the real training run.
    """
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=build_transforms(train=True))
    valid_dataset = datasets.ImageFolder(str(VALID_DIR), transform=build_transforms(train=False))

    class_names = train_dataset.classes

    # Persist class names now — the API (Stage 4) needs this exact list, in this
    # exact order, to translate model output indices back into human-readable labels.
    CLASS_NAMES_PATH.parent.mkdir(exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    if subset_size is not None:
        train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:subset_size].tolist())
        valid_dataset = Subset(valid_dataset, torch.randperm(len(valid_dataset))[:subset_size].tolist())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, class_names
