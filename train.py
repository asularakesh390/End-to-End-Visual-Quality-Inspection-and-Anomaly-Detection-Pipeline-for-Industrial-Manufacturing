import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.backbone import FeatureExtractor
from src.dataset import IndustrialDefectDataset

def train(data_dir, output_model_path, batch_size=16):
    print(f"[INFO] Initializing Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on computational device: {device}")

    # Dataset & Loader
    train_dataset = IndustrialDefectDataset(root_dir=data_dir, is_train=True)
    if len(train_dataset) == 0:
        raise ValueError(f"No valid images found in {data_dir}/good. Please verify your data directory.")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Feature Extractor
    model = FeatureExtractor(pretrained=True).to(device)
    model.eval()

    features_list = []

    print("[INFO] Extracting normal feature distributions...")
    with torch.no_grad():
        for images, _, _ in train_loader:
            images = images.to(device)
            feats = model(images)
            # Pool spatial dimensions to produce global feature vector
            pooled_feats = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).squeeze(-1).squeeze(-1) for f in feats]
            concat_feats = torch.cat(pooled_feats, dim=1)
            features_list.append(concat_feats.cpu().numpy())

    features = np.vstack(features_list)
    
    # Calculate Gaussian stats for normal baseline class
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False) + 1e-5 * np.eye(features.shape[1])
    inv_cov = np.linalg.pinv(cov)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open(output_model_path, 'wb') as f:
        pickle.dump({'mean': mean, 'inv_cov': inv_cov}, f)

    print(f"[SUCCESS] Model baseline saved successfully to: {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Defect Detection Baseline Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--output", type=str, default="weights/model_baseline.pkl", help="Output path for weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train(args.data_dir, args.output, args.batch_size)
