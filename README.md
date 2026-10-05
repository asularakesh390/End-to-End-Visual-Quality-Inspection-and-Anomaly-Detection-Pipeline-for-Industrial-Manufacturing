Automated Surface Defect & Anomaly Detection SystemAn end-to-end, command-line executable Computer Vision pipeline designed for visual quality control and anomaly detection in industrial manufacturing. The system leverages pre-trained deep feature representations and statistical distance modeling (Mahalanobis Distance) to detect defects without requiring labeled defective data during training.Table of ContentsProject OverviewKey FeaturesRepository StructureEnvironment Setup & InstallationDataset FormattingUsage & Execution Guide1. Model Training2. Model Evaluation3. Single Image InferenceModel Architecture & MethodologyLicenseProject OverviewQuality inspection in manufacturing often faces a data imbalance problem: defective samples are rare, unpredictable, and costly to label. This project implements an unsupervised anomaly detection framework using deep convolutional feature extraction (ResNet-18).By modeling the statistical distribution of normal ("good") product features, the pipeline calculates an Anomaly Score for unseen test items, flags defective items, and outputs visual annotations—all through terminal-executable commands.Key FeaturesCLI First Design: Fully operable via terminal interface with standard arguments (argparse), making it ideal for automated grading pipelines and headless servers.Unsupervised Learning: Requires only non-defective (normal) images during training.Deep Feature Embeddings: Extracts multi-scale visual features using a PyTorch pre-trained backbone.Robust Anomaly Scoring: Uses Mahalanobis distance covariance modeling for fast, accurate defect identification.Automated Visualization: Annotates input images with prediction results and calculated confidence scores.Repository Structure.
├── data/
│   ├── train/
│   │   └── good/              # Training images (normal samples only)
│   └── test/
│       ├── good/              # Test set (normal samples)
│       └── defective/         # Test set (defective samples)
├── models/
│   ├── __init__.py
│   └── backbone.py            # Feature extraction network definition
├── src/
│   ├── __init__.py
│   └── dataset.py             # PyTorch Dataset and preprocessing loaders
├── weights/
│   └── .gitkeep               # Directory for trained model artifacts
├── evaluate.py                # Quantitative batch evaluation script
├── infer.py                   # Single image inference script
├── README.md                  # Project documentation and execution instructions
├── requirements.txt           # Pinned library dependencies
└── train.py                   # Main model training script
Environment Setup & InstallationPrerequisitesPython: 3.9 or 3.10OS: Linux, macOS, or Windows (Terminal / PowerShell / Bash)Step 1: Clone the Repositorygit clone https://github.com/<your-github-username>/<your-repo-name>.git
cd <your-repo-name>
Step 2: Create a Virtual Environment# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
Step 3: Install Dependenciespip install --upgrade pip
pip install -r requirements.txt
Dataset FormattingOrganize your dataset inside the data/ directory according to the structure below:data/
├── train/
│   └── good/                  # Place clean/normal training images here (.png, .jpg)
└── test/
    ├── good/                  # Place normal test images here
    └── defective/             # Place defective test images here
Note: Subdirectories can contain common image formats (.png, .jpg, .jpeg, .bmp).Usage & Execution GuideAll execution commands are run directly from the root directory of the project.1. Model TrainingTrain the baseline distribution model on normal manufacturing samples:python train.py --data_dir data/train --output weights/model_baseline.pkl --batch_size 16
Parameters:--data_dir: Path to training directory containing the good/ subfolder.--output: Filepath where the trained model weights/statistics will be saved.--batch_size: (Optional) Batch size for feature extraction (Default: 16).2. Model EvaluationEvaluate model accuracy, ROC-AUC score, Precision, and Recall across test samples:python evaluate.py --test_dir data/test --model weights/model_baseline.pkl
Parameters:--test_dir: Path to test directory containing good/ and defective/ subfolders.--model: Path to saved model weights (.pkl).3. Single Image InferenceRun defect analysis on a single image to generate a prediction and annotated visual output:python infer.py --image data/test/defective/sample_01.png --model weights/model_baseline.pkl --threshold 15.0
Parameters:--image: Path to target input image.--model: Path to trained model file.--threshold: (Optional) Sensitivity threshold for flagging anomalies (Default: 15.0).Output: The script prints the anomaly score to the console and outputs an annotated image output_prediction.png with visual tags (NORMAL or DEFECTIVE).Model Architecture & MethodologyFeature Extraction: Images are passed through an ImageNet-pretrained ResNet-18 network. Intermediate tensor representations from multiple convolutional stages are pooled and concatenated to form a comprehensive visual embedding vector.Distribution Modeling: During training, the multivariate Gaussian distribution (mean vector $\mu$ and inverted covariance matrix $\Sigma^{-1}$) of normal features is computed.Distance Scoring: During inference, test sample embeddings $x$ are scored using the Mahalanobis Distance:
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
Samples with $D_M(x) > \text{threshold}$ are classified as defective.LicenseDistributed under the MIT License. See LICENSE for more information.
