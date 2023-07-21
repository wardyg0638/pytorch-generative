import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class FabricDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_fnames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_fnames[idx])
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, 0