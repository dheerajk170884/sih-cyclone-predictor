import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CycloneDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load satellite image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Target: Cyclone category (Class ID) or Wind Speed
        label = int(row['category_label'])
        
        return image, torch.tensor(label, dtype=torch.long)