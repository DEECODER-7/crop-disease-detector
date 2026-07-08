"""
Central configuration for training and inference.

Keeping these values in one place (instead of scattered magic numbers across
files) means Stage 4 (the API) and Stage 8 (the UI) can import the exact same
constants the model was trained with — critical, because a mismatch between
training-time preprocessing and inference-time preprocessing is one of the
most common silent bugs in production ML systems.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Image size ResNet18 was originally trained on. We resize all inputs to this
# at both training and inference time — must match exactly or accuracy drops.
IMAGE_SIZE = 224

# ImageNet normalization stats — required because we're using ImageNet-pretrained
# weights. The pretrained conv layers expect inputs normalized this way.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
NUM_EPOCHS = 5          # transfer learning needs far fewer epochs than training from scratch
LEARNING_RATE = 1e-4
NUM_WORKERS = 2         # dataloader parallelism; lower this to 0 if you hit issues on Windows

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.json"
