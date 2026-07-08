"""
Training script for the crop disease classifier.

Usage:
    # Smoke test first — confirms the whole pipeline works, takes ~1-2 min on CPU
    python -m src.models.train --subset 200 --epochs 1

    # Real training run (do this on Colab/GPU, not your laptop CPU)
    python -m src.models.train --epochs 5
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.architecture import build_model
from src.models.config import BEST_MODEL_PATH, CHECKPOINT_DIR, LEARNING_RATE, NUM_EPOCHS
from src.models.dataset import get_dataloaders


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Runs one full pass over `loader`. Shared between train and validation to
    avoid duplicating (and accidentally diverging) the forward-pass logic —
    the only difference is whether we backprop and whether we call .train()/.eval().
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only this many images for train/valid (for fast smoke-testing)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: training on CPU. This is fine for a smoke test (--subset), "
              "but a full run will be slow — consider Google Colab (free GPU) for that.")

    print("Loading data...")
    train_loader, valid_loader, class_names = get_dataloaders(subset_size=args.subset)
    print(f"Classes: {len(class_names)} | Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    print("Building model (ResNet18, frozen backbone, new final layer)...")
    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    # Only the unfrozen params (the new final layer) have requires_grad=True,
    # so this optimizer is only actually updating that layer, as intended.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_valid_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "valid_acc": valid_acc,
            }, BEST_MODEL_PATH)
            print(f"  -> New best model saved (valid_acc={valid_acc:.4f}) to {BEST_MODEL_PATH}")

    print(f"\nTraining complete. Best validation accuracy: {best_valid_acc:.4f}")
    print(f"Best model checkpoint: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
