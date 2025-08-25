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

        # 遍历掩膜目录，仅保留存在相应掩膜的图像
        for mask_filename in sorted(os.listdir(self.mask_dir)):
            if mask_filename.endswith('.nii.gz'):
                img_path = os.path.join(self.img_dir, mask_filename)
                mask_path = os.path.join(self.mask_dir, mask_filename)

                # 确保图像和掩膜都存在
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.img_files.append(img_path)
                    self.mask_files.append(mask_path)
                else:
                    print(f"Missing one of the files: {img_path} or {mask_path}")

        # 数据增强的组合
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

        # 测试模式下的标准化
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

            # 转换为 PyTorch 张量
            img = img_tensor.data.squeeze(0).permute(2, 0, 1)  # 从 [C, W, H, D] 变成 [D, W, H]
            mask = mask_tensor.data.squeeze(0).permute(2, 0, 1)  # 从 [C, W, H, D] 变成 [D, W, H]

            # 转换 mask 为 one-hot 编码
            mask = self.to_one_hot_3d(mask, n_classes=2)
            return img, mask

        except (RuntimeError, Exception) as e:
            print(f"Skipping sample {idx} due to error:")
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")
            print(f"Error message: {str(e)}")

            # 返回占位数据而不是None
            depth, width, height = 256, 512, 512  # 根据您的数据调整这些尺寸
            img_placeholder = torch.zeros((depth, width, height))
            mask_placeholder = torch.zeros((2, depth, width, height))

            return img_placeholder, mask_placeholder

    @staticmethod
    def to_one_hot_3d(tensor, n_classes=2):
        # 确保张量值在 0 到 (n_classes-1) 的范围内
        tensor = torch.clamp(tensor, 0, n_classes - 1)
        tensor = tensor.to(torch.long)

        # 创建 one-hot 张量
        one_hot = torch.zeros((n_classes, *tensor.shape), dtype=torch.float32)

        # 生成 one-hot 编码
        one_hot = one_hot.scatter_(0, tensor.unsqueeze(0), 1)

        return one_hot


if __name__ == "__main__":
    # 设置数据目录
    img_dir = "D:\\NNIP\\LungSeg\\dataset\\LUNA16\\ResizeData"
    mask_dir = "D:\\NNIP\\LungSeg\\dataset\\LUNA16\\ResizeMask"

    # 创建数据集实例
    dataset = niiGzDataset(img_dir, mask_dir, _mode='train')

    # 打印数据集长度
    print(f"Total samples in dataset: {len(dataset)}")

    # 测试读取前几个样本
    for i in range(0, 5):  # 读取前5个样本
        img, mask = dataset[i]
        print(f"Sample {i}:")
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Image values range: [{img.min()}, {img.max()}]")
        print(f"Mask unique values: {torch.unique(mask)}")