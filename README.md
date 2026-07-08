# Crop Disease Detector — End-to-End ML Engineering Project

Production-style ML system: data pipeline → PyTorch CNN (transfer learning) →
FastAPI serving → Docker → GitHub Actions CI/CD → AWS deployment → Streamlit UI.

Built to directly demonstrate the skills required for AI/ML Engineer (Remote) roles:
Python, PyTorch, data preprocessing/feature engineering, model evaluation,
cloud deployment (AWS), MLOps/CI-CD, and translating a business problem into
a working technical system.

## Business problem
Farmers lose significant crop yield to diseases that go undetected until visible
damage has already spread. A model that classifies a photographed leaf as
healthy or diseased (and which disease) lets a farmer act early — this is the
same "vision model on constrained hardware, real-world noisy input" problem
space as precision-agriculture robotics, just applied to a new dataset.

## Dataset
PlantVillage (via the Kaggle "New Plant Diseases Dataset" mirror) — 38 classes,
~54,000 images, healthy + diseased leaf images across 14 crop species.

## Project stages
- [x] Stage 1: Project scaffold + data pipeline
- [ ] Stage 2: Model training (PyTorch, transfer learning w/ ResNet18)
- [ ] Stage 3: Evaluation (precision/recall/F1, confusion matrix, per-class report)
- [ ] Stage 4: FastAPI inference service
- [ ] Stage 5: Dockerize
- [ ] Stage 6: CI/CD with GitHub Actions
- [ ] Stage 7: Deploy to AWS (free tier)
- [ ] Stage 8: Streamlit UI
- [ ] Stage 9: Basic monitoring/logging

## Folder structure
```
crop-disease-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data/
│   │   └── download_data.py      # fetches + organizes the dataset
│   ├── models/                    # (Stage 2) model architecture, training loop
│   └── api/                       # (Stage 4) FastAPI app
├── tests/                          # unit tests (Stage 6 wires these into CI)
├── notebooks/                      # exploratory work only — not the deliverable
└── data/                           # gitignored, created by download_data.py
```

## Stage 1 setup (do this now)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a Kaggle API token (needed once):
   - Go to kaggle.com → Account → "Create New API Token" → downloads `kaggle.json`
   - Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows)
4. Run the data download script:
   ```bash
   python src/data/download_data.py
   ```
   This downloads the dataset and organizes it into `data/train`, `data/valid`,
   `data/test` folders, one subfolder per class.

## Why we're doing it this way (read this before you run anything)

- **Kaggle API instead of a manual zip download**: reproducibility. Anyone
  (including an interviewer asking you to walk through your process) should be
  able to clone your repo and get the exact same data without you emailing them
  a zip file. This is a small thing that signals "engineer" vs "notebook hobbyist."
- **Separate `data/` folder, gitignored**: datasets don't belong in git history —
  they bloat the repo and Git isn't built for large binary diffs. This is standard
  practice and worth mentioning if asked about repo hygiene in an interview.
- **`src/` package layout**: keeps import paths clean and makes the codebase look
  like a real package rather than a pile of scripts, which matters when you get
  to Stage 4 (the API imports from `src.models`).

Once you've run the download script and can see `data/train/<class_name>/*.jpg`
folders populated, tell me and we'll move to Stage 2 (model training).
