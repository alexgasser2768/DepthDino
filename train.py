import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import os, logging

from model import ConvNeXtDepthModel
from dataloader import DepthDataset
from losses import SILogLoss, DistillationLoss

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename="log", level=logging.INFO)


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
            loss = criterion(preds, t_depths)  # Evaluation against teacher
            running_loss += loss.item()

    return running_loss / len(loader)


if __name__ == "__main__":
    # Settings
    DATA_DIR = "data/unlabeled2017/"
    BATCH_SIZE = 125
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    PATCH_SIZE = 160
    NUM_WORKERS = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("weights", exist_ok=True)

    # 1. Setup Student Model
    logger.info(f"Initializing student model on {DEVICE}...")
    model = ConvNeXtDepthModel(arch='convnext_tiny.dinov3_lvd1689m', mlp_weights_path="weights/best_model.pth")
    model.to(DEVICE)

    # 2. Setup Data
    dataset = DepthDataset(DATA_DIR, patch_width=PATCH_SIZE, patch_height=PATCH_SIZE, transforms=model.transforms, device=DEVICE)

    # Simple split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Setup Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    distill_loss = DistillationLoss()
    val_criterion = SILogLoss()

    # 4. Training Loop
    logger.info("Starting training with cached distillation...")

    best_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_one_epoch(model, train_loader, optimizer, distill_loss, DEVICE)
        val_loss = validate(model, val_loader, val_criterion, DEVICE)

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/best_model.pth")
            logger.info(f"--> New best model saved.")

        # Save checkpoint
        torch.save(model.state_dict(), f"weights/depth_model_epoch_{epoch + 1}.pth")

    logger.info("Training complete.")
