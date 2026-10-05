import argparse
import pickle
import cv2
import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from models.backbone import FeatureExtractor

def predict(image_path, model_path, threshold=15.0, save_output=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load feature stats
    with open(model_path, 'rb') as f:
        stats = pickle.load(f)
    mean, inv_cov = stats['mean'], stats['inv_cov']

    # Preprocess image
    image_orig = cv2.imread(image_path)
    if image_orig is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
        
    image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    # Feature extraction
    model = FeatureExtractor(pretrained=True).to(device)
    model.eval()
    
    with torch.no_grad():
        feats = model(tensor)
        pooled_feats = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).squeeze(-1).squeeze(-1) for f in feats]
        feat_vec = torch.cat(pooled_feats, dim=1).cpu().numpy().flatten()

    # Calculate Anomaly Score (Mahalanobis Distance)
    dist = mahalanobis(feat_vec, mean, inv_cov)
    is_anomaly = dist > threshold

    status_str = "DEFECTIVE" if is_anomaly else "NORMAL"
    color = (0, 0, 255) if is_anomaly else (0, 255, 0)

    print(f"\n========================================")
    print(f" Image: {image_path}")
    print(f" Anomaly Score: {dist:.4f}")
    print(f" Threshold:     {threshold}")
    print(f" Decision:      [{status_str}]")
    print(f"========================================\n")

    if save_output:
        annotated = image_orig.copy()
        cv2.putText(annotated, f"Status: {status_str} (Score: {dist:.1f})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        out_path = "output_prediction.png"
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Annotated visualization output saved to '{out_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on Single Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="weights/model_baseline.pkl", help="Path to baseline model")
    parser.add_argument("--threshold", type=float, default=15.0, help="Anomaly decision threshold")
    args = parser.parse_args()

    predict(args.image, args.model, args.threshold)
