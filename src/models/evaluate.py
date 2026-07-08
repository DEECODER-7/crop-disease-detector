"""
Evaluates the trained model on the validation set and produces:
  1. Overall accuracy, macro/weighted precision, recall, F1
  2. A full per-class classification report (sklearn)
  3. A confusion matrix plot saved to checkpoints/confusion_matrix.png

Why this matters more than the single "valid_acc" number from training:
accuracy alone hides problems. With 38 classes, a model can hit 90% overall
accuracy while being genuinely bad at 3-4 specific classes that look visually
similar to each other (e.g. two different leaf-spot diseases). Precision/recall
per class surfaces exactly which classes are confused with which — this is
the kind of analysis an interviewer is actually checking for when they ask
"how did you evaluate your model."

Usage:
    python -m src.models.evaluate
"""

import json

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.models.architecture import build_model
from src.models.config import BEST_MODEL_PATH, CHECKPOINT_DIR
from src.models.dataset import get_dataloaders
from src.models.train import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("Loading validation data...")
    _, valid_loader, class_names = get_dataloaders()

    print(f"Loading checkpoint from {BEST_MODEL_PATH} ...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    saved_class_names = checkpoint["class_names"]

    if saved_class_names != class_names:
        raise ValueError(
            "Class name order in checkpoint doesn't match current dataset's class order. "
            "This usually means the dataset folder changed between training and evaluation. "
            "Re-run training, or make sure you're evaluating against the same data/train folder."
        )

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 70)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    print(report)

    # Save the report as text too, so it can be pasted into a README or resume bullet
    report_path = CHECKPOINT_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Full report saved to {report_path}")

    print("\nBuilding confusion matrix plot...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Crop Disease Classifier")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = CHECKPOINT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")

    # Identify the most-confused class pairs — this is the part worth
    # mentioning in an interview: not just "here's a confusion matrix" but
    # "here are specifically the classes the model struggles to tell apart."
    print("\nTop confused class pairs (excluding correct predictions):")
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    for count, true_class, pred_class in confusions[:10]:
        print(f"  {count:4d}x  true={true_class}  ->  predicted={pred_class}")


if __name__ == "__main__":
    main()
