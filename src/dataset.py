import os
import cv2
import torch
from torch.utils.data import Dataset

class IndustrialDefectDataset(Dataset):
    """
    Custom Dataset loader for industrial surface images.
    Supports directory-based structure: dataset/good/ and dataset/defective/
    """
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()

    def _load_dataset(self):
        good_dir = os.path.join(self.root_dir, 'good')
        if os.path.exists(good_dir):
            for fname in os.listdir(good_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(good_dir, fname))
                    self.labels.append(0)  # 0 for Normal/Good
                    
        if not self.is_train:
            defective_dir = os.path.join(self.root_dir, 'defective')
            if os.path.exists(defective_dir):
                for fname in os.listdir(defective_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.image_paths.append(os.path.join(defective_dir, fname))
                        self.labels.append(1)  # 1 for Anomaly/Defective

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # Transpose image to (C, H, W) and normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label, img_path
