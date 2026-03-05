import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
from tqdm import tqdm
import os, glob, logging, PIL.Image

from model import ConvNeXtDepthModel, PREPROCESS
from losses import SILogLoss, DistillationLoss
from depth_anything_3.api import DepthAnything3

PATCH_WIDTH = 224
PATCH_HEIGHT = 224
NUM_WORKERS = 16

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename="log", level=logging.INFO)

class DepthDataset(Dataset):
    def __init__(self, data_dir, teacher=None, device="cpu"):
        self.data_dir = data_dir
        self.teacher = teacher
        self.device = device

        # Find all color files
        self.color_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        logger.info(f"Found {len(self.color_files)} images in {data_dir}.")

        self.student_augment = T.Compose([
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
                raise RuntimeError(f"Teacher model required to generate missing cache file for {color_path}")
            
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
        
        # 3. Apply Augmentations
        # A. Light Visual Augmentations (Student Only)
        aug_image_pil = self.student_augment(image_pil)
        
        # B. Geometric Augmentations (Applied to both image and teacher labels)
        # Convert labels to tensors for easier manipulation
        depth_tensor = torch.from_numpy(t_depth).unsqueeze(0) # [1, H, W]
        conf_tensor = torch.from_numpy(t_conf).unsqueeze(0)   # [1, H, W]
        
        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            aug_image_pil = TF.hflip(aug_image_pil)
            depth_tensor = TF.hflip(depth_tensor, dims=[-1])
            conf_tensor = TF.hflip(conf_tensor, dims=[-1])

        # Random Vertical Flip
        if np.random.rand() > 0.5:
            aug_image_pil = TF.vflip(aug_image_pil)
            depth_tensor = TF.vflip(depth_tensor, dims=[-2])
            conf_tensor = TF.vflip(conf_tensor, dims=[-2])

        # 4. Crop to Patch Size
        w, h = aug_image_pil.size
        top = np.random.randint(0, h - PATCH_HEIGHT + 1)
        left = np.random.randint(0, w - PATCH_WIDTH + 1)

        # Crop PIL image
        aug_image_pil = aug_image_pil.crop((left, top, left + PATCH_WIDTH, top + PATCH_HEIGHT))

        # Crop Tensors
        depth_tensor = depth_tensor[:, top:top+PATCH_HEIGHT, left:left+PATCH_WIDTH]
        conf_tensor = conf_tensor[:, top:top+PATCH_HEIGHT, left:left+PATCH_WIDTH]

        # 5. Final Preprocessing for Student
        image_tensor = PREPROCESS(aug_image_pil)

        return image_tensor, depth_tensor, conf_tensor


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, t_depths, t_confs in loader:
        images = images.to(device)
        t_depths = t_depths.to(device)
        t_confs = t_confs.to(device)

        # Forward
        preds = model(images)
        
        # Compute Distillation Loss (using teacher's pseudo-labels)
        loss = criterion(preds, t_depths, t_confs)
        
        if torch.isnan(loss):
            logger.warning("NaN loss detected!")
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, t_depths, t_confs in loader:
            images = images.to(device)
            t_depths = t_depths.to(device)
            t_confs = t_confs.to(device)

            preds = model(images)
            loss = criterion(preds, t_depths) # Evaluation against teacher
            running_loss += loss.item()
    return running_loss / len(loader)


if __name__ == "__main__":
    # Settings
    DATA_DIR = "data/"
    BATCH_SIZE = 64 
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Teacher Model (Depth Anything V3)
    logger.info("Loading Teacher Model: Depth Anything V3 Small...")
    teacher = DepthAnything3.from_pretrained("depth-anything/da3-small")
    teacher.to(DEVICE)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # 2. Setup Data
    dataset = DepthDataset(DATA_DIR, teacher=teacher, device=DEVICE)

    # Simple split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # First epoch might be slow due to caching, num_workers=0 avoids CUDA complexity
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 3. Setup Student Model
    logger.info(f"Initializing student model on {DEVICE}...")
    model = ConvNeXtDepthModel(arch='convnext_tiny.dinov3_lvd1689m', mlp_weights_path="weights/decoder/best_model.pth")
    model.to(DEVICE)

    # 4. Setup Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    distill_loss = DistillationLoss()
    val_criterion = SILogLoss()

    # 5. Training Loop
    logger.info("Starting training with cached distillation...")

    best_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_one_epoch(model, train_loader, optimizer, distill_loss, DEVICE)
        val_loss = validate(model, val_loader, val_criterion, DEVICE)

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/decoder/best_model.pth")
            logger.info(f"--> New best model saved.")

        # Save checkpoint
        torch.save(model.state_dict(), f"weights/decoder/depth_model_epoch_{epoch + 1}.pth")

    logger.info("Training complete.")
