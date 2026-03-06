import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import os, glob, logging, PIL.Image
import numpy as np
from depth_anything_3.api import DepthAnything3

logger = logging.getLogger(__name__)


class DepthDataset(Dataset):
    def __init__(self, data_dir, patch_width=224, patch_height=224, transforms=T.ToTensor(), device="cpu"):
        self.data_dir = data_dir
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.teacher = None  # Will be loaded on demand
        self.device = device

        # Find all color files
        self.color_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        logger.info(f"Found {len(self.color_files)} images in {data_dir}.")

        self.student_augment = T.Compose([
            transforms,
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        ])

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        color_path = self.color_files[idx]
        cache_path = color_path.replace(".jpg", ".da3_cache.npz")

        # 1. Load or Generate Teacher Pseudo-Labels
        if not os.path.exists(cache_path):
            if self.teacher is None:
                self.teacher = DepthAnything3.from_pretrained("depth-anything/da3-small")
                self.teacher.to(self.device)
                self.teacher.eval()
                for param in self.teacher.parameters():
                    param.requires_grad = False

            # Run teacher on original image
            with torch.no_grad():
                # We use PIL for the teacher's inference method as it expects paths or PIL/numpy
                prediction = self.teacher.inference([color_path])

                # Extract depth and confidence (N=1)
                t_depth = prediction.depth[0].astype(np.float32)
                t_conf = prediction.conf[0].astype(np.float32) if prediction.conf is not None else np.ones_like(t_depth)

                # Save to cache
                np.savez_compressed(cache_path, depth=t_depth, conf=t_conf)

        # Load from cache
        try:
            cache = np.load(cache_path)
            t_depth = cache['depth']
            t_conf = cache['conf']
        except Exception as e:
            logger.error(f"Error loading cache {cache_path}: {e}")
            # Fallback if cache is corrupted
            os.remove(cache_path)
            return self.__getitem__(idx)

        # 2. Load Image
        image_pil = PIL.Image.open(color_path).convert("RGB")

        # Interpolate image to match teacher output size if needed
        if image_pil.size != (t_depth.shape[1], t_depth.shape[0]):
            image_pil = image_pil.resize((t_depth.shape[1], t_depth.shape[0]), resample=PIL.Image.BILINEAR)

        # 3. Apply Augmentations
        # A. Light Visual Augmentations (Student Only)
        image_tensor = self.student_augment(image_pil)

        # B. Geometric Augmentations (Applied to both image and teacher labels)
        # Convert labels to tensors for easier manipulation
        depth_tensor = torch.from_numpy(t_depth).unsqueeze(0) # [1, H, W]
        conf_tensor = torch.from_numpy(t_conf).unsqueeze(0)   # [1, H, W]

        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            image_tensor = TF.hflip(image_tensor)
            depth_tensor = TF.hflip(depth_tensor)
            conf_tensor = TF.hflip(conf_tensor)

        # Random Vertical Flip
        if np.random.rand() > 0.5:
            image_tensor = TF.vflip(image_tensor)
            depth_tensor = TF.vflip(depth_tensor)
            conf_tensor = TF.vflip(conf_tensor)

        # 4. Crop to Patch Size
        w, h = image_tensor.shape[2], image_tensor.shape[1]
        top = np.random.randint(0, h - self.patch_height + 1)
        left = np.random.randint(0, w - self.patch_width + 1)

        # Crop Tensors
        image_tensor = image_tensor[:, top:top + self.patch_height, left:left + self.patch_width]
        depth_tensor = depth_tensor[:, top:top + self.patch_height, left:left + self.patch_width]
        conf_tensor = conf_tensor[:, top:top + self.patch_height, left:left + self.patch_width]

        return image_tensor, depth_tensor, conf_tensor
