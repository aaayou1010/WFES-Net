import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchio as tio


class niiGzDataset(Dataset):
    def __init__(self, img_dir, mask_dir, _mode='train', cache_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = _mode

        self.img_files = []
        self.mask_files = []

        for mask_filename in sorted(os.listdir(self.mask_dir)):
            if mask_filename.endswith('.nii.gz'):
                img_path = os.path.join(self.img_dir, mask_filename)
                mask_path = os.path.join(self.mask_dir, mask_filename)
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.img_files.append(img_path)
                    self.mask_files.append(mask_path)
                else:
                    print(f"Missing one of the files: {img_path} or {mask_path}")

        self.transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)),
            tio.RandomNoise(std=(0, 0.05)),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=(-10, 10),
                translation=(-5, 5)
            ),
        ])

        self.norm = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99))

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        try:
            img_tensor = tio.ScalarImage(img_path)
            mask_tensor = tio.LabelMap(mask_path)

            subject = tio.Subject(
                img=img_tensor,
                mask=mask_tensor
            )

            if self.mode == 'train':
                print('训练模式')
                transformed = self.transform(subject)
                img_tensor = transformed.img
                mask_tensor = transformed.mask
            else:
                subject.img = self.norm(subject.img)
                img_tensor = subject.img
                mask_tensor = subject.mask

            img = img_tensor.data.squeeze(0).permute(2, 0, 1) 
            mask = mask_tensor.data.squeeze(0).permute(2, 0, 1)

            mask = self.to_one_hot_3d(mask, n_classes=2)
            return img, mask

        except (RuntimeError, Exception) as e:
            print(f"Skipping sample {idx} due to error:")
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")
            print(f"Error message: {str(e)}")

            depth, width, height = 256, 512, 512  
            img_placeholder = torch.zeros((depth, width, height))
            mask_placeholder = torch.zeros((2, depth, width, height))

            return img_placeholder, mask_placeholder

    @staticmethod
    def to_one_hot_3d(tensor, n_classes=2):
        tensor = torch.clamp(tensor, 0, n_classes - 1)
        tensor = tensor.to(torch.long)

        one_hot = torch.zeros((n_classes, *tensor.shape), dtype=torch.float32)

        one_hot = one_hot.scatter_(0, tensor.unsqueeze(0), 1)

        return one_hot
