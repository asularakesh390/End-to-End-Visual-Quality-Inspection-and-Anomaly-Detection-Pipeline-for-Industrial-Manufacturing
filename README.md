# End-to-End-Visual-Quality-Inspection-and-Anomaly-Detection-Pipeline-for-Industrial-Manufacturing

cv-manufacturing-defect-detection/
├── data/
│   ├── raw/                  # Place raw images here
│   └── processed/            # Preprocessed and normalized data
├── models/
│   ├── backbone.py           # Feature extractor (ResNet/EfficientNet)
│   └── anomaly_detector.py   # Mahalanobis / PatchCore / Autoencoder modules
├── notebooks/
│   └── exploration.ipynb     # Initial EDA and data visualization
├── src/
│   ├── dataset.py            # Custom PyTorch Dataset & DataLoaders
│   ├── utils.py              # Metrics, visualizers, image I/O
│   └── augmentations.py     # Albumentations pipeline
├── weights/
│   └── .gitkeep              # Directory to store trained checkpoints
├── train.py                  # Main training entry point (CLI)
├── infer.py                  # Pipeline execution / inference script (CLI)
├── evaluate.py               # Quantitative evaluation script (CLI)
├── README.md                 # Setup, configuration, and execution guide
├── requirements.txt          # Explicitly pinned python dependencies
└── .gitignore                # Excludes checkpoints, raw data, and cache
