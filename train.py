import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm
import os, glob, cv2, logging

from model import ConvNeXtDepthModel, PREPROCESS
from losses import DepthLoss, SILogLoss, GradientMatchingLoss, VirtualNormalLoss

PATCH_WIDTH = 224
PATCH_HEIGHT = 224
NUM_WORKERS = 16

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename="log", level=logging.INFO)

LAMBDA = 5

class DepthDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to directory containing .hdf5 files
        """
        self.data_dir = data_dir

        # Find all color files
        self.color_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.data_pairs = []

        # Pair them with depth files
        for color_path in self.color_files:
            depth_path = color_path.replace(".jpg", ".png")

            if os.path.exists(depth_path):
                self.data_pairs.append((color_path, depth_path))
            else:
                logger.warning(f"Depth file missing for {color_path}, skipping.")

        logger.info(f"Found {len(self.data_pairs)} valid image-depth pairs.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        color_path, depth_path = self.data_pairs[idx]

        # --- Load RGB ---
        image = cv2.imread(color_path, cv2.IMREAD_COLOR_RGB).astype(np.float32) / 255.0
        image_tensor = PREPROCESS(image)

        # --- Load Depth ---
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth[depth >= 65_535] = 0
        if "cm" in depth_path:
            depth /= 100.0  # Convert cm to meters
        else:
            depth /= 1000.0  # Convert millimeters to meters

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # Convert to Tensor [1, H, W]

        # Crop a random region that is 352x480 (352 is divisible by 32, required by the model)
        h, w = depth_tensor.shape[1:]
        top = np.random.randint(0, h - PATCH_HEIGHT + 1)
        left = np.random.randint(0, w - PATCH_WIDTH + 1)
        depth_tensor = depth_tensor[:, top:top+PATCH_HEIGHT, left:left+PATCH_WIDTH]
        image_tensor = image_tensor[:, top:top+PATCH_HEIGHT, left:left+PATCH_WIDTH]

        if np.random.rand() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[-1])
            depth_tensor = torch.flip(depth_tensor, dims=[-1])

        return image_tensor, depth_tensor


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        # Forward
        preds = model(images)
        loss = criterion(preds, targets)
        if torch.isnan(loss):
            logger.warning("NaN loss detected!")
            continue  # Skip this batch

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)

            loss = criterion(preds, targets)
            running_loss += loss.item()
    return running_loss / len(loader)


if __name__ == "__main__":
    # Settings
    DATA_DIR = "data/"
    CONFIG_FILE = "weights/tiny/config.json"
    WEIGHTS_FILE = "weights/tiny/model.safetensors"
    BATCH_SIZE = 100
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup Data
    dataset = DepthDataset(DATA_DIR)

    # Simple split (e.g., last 20% for validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # Setup Model
    logger.info(f"Initializing model on {DEVICE}...")
    model = ConvNeXtDepthModel(CONFIG_FILE, WEIGHTS_FILE, mlp_weights_path="weights/decoder/best_model.pth")
    model.to(DEVICE)

    # Setup Optimizer & Loss
    # Only optimize the decoder parameters (backbone is frozen inside the model class)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    depth_loss = DepthLoss()
    silog_loss = SILogLoss()
    gradient_loss = GradientMatchingLoss()
    vn_loss = VirtualNormalLoss()

    criterion = lambda preds, targets: depth_loss(preds, targets) + LAMBDA * silog_loss(preds, targets) + gradient_loss(preds, targets) + vn_loss(preds, targets)

    # Training Loop
    logger.info("Starting training...")

    best_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/decoder/best_model.pth")
            logger.info(f"--> New best model saved with Val Loss: {val_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"weights/decoder/depth_model_epoch_{epoch + 1}.pth")

    logger.info("Training complete.")
