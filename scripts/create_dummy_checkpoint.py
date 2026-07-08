"""
Creates a small, randomly-initialized checkpoint purely so CI can test the
API's code (request/response handling, validation, error cases) WITHOUT
needing the real ~45MB trained model or the multi-GB dataset in the repo.

This is standard practice: CI should test that your code works, not
re-validate model accuracy on every push (that's what evaluate.py is for,
run manually/separately). Never used for real predictions — only for
automated testing.

Usage:
    python scripts/create_dummy_checkpoint.py
"""

import json

import torch

from src.models.architecture import build_model
from src.models.config import BEST_MODEL_PATH, CHECKPOINT_DIR, CLASS_NAMES_PATH

# Small number of fake classes — enough to exercise the code paths
# (top-3 predictions, class name lookup, etc.) without downloading anything.
DUMMY_CLASS_NAMES = [f"dummy_class_{i}" for i in range(38)]


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    model = build_model(num_classes=len(DUMMY_CLASS_NAMES))

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": DUMMY_CLASS_NAMES,
        "valid_acc": 0.0,  # meaningless — this model was never trained
    }, BEST_MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(DUMMY_CLASS_NAMES, f)

    print(f"Dummy checkpoint created at {BEST_MODEL_PATH} (for CI testing only)")


if __name__ == "__main__":
    main()