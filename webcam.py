import torch
import cv2
import numpy as np
from PIL import Image
import time
import logging

from agent import ConvNeXtDepthModel, PREPROCESS

logger = logging.getLogger(__name__) 
logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename="log", level=logging.INFO)


def run_webcam():
    # --- Configuration ---
    CONFIG_FILE = "weights/tiny/config.json"
    WEIGHTS_FILE = "weights/tiny/model.safetensors" # Or "depth_model_epoch_X.pth" if you trained it
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_SIZE = (224, 224) # Size model expects
    
    logger.info(f"Loading model on {DEVICE}...")
    
    # Initialize model
    model = ConvNeXtDepthModel(CONFIG_FILE, WEIGHTS_FILE, mlp_weights_path="weights/decoder/best_model.pth")
    model.to(DEVICE)
    model.eval()

    # Start Webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default camera
    if not cap.isOpened():
        logger.error("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    with torch.no_grad():
        while True:
            start_time = time.time()
            
            # Read Frame
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Prepare Input
            # OpenCV is BGR, Model needs RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_tensor = PREPROCESS(img_rgb).unsqueeze(0).to(DEVICE)

            # 2. Inference
            depth_map = model(input_tensor) # Output: [1, 1, 224, 224]

            # 3. Post-processing for Visualization
            # Remove batch dim and move to CPU
            depth_np = depth_map.squeeze().cpu().numpy()
            
            # Normalize to 0-255 for visualization
            # (We use dynamic normalization based on min/max in the current frame)
            d_min = depth_np.min()
            d_max = depth_np.max()
            
            if d_max - d_min > 1e-5:
                depth_norm = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_np)

            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # Apply colormap (Magma or Inferno look good for depth)
            overlayed = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0.3, depth_uint8, 0.7, 0)
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


            # 4. Display Result
            # Stack images horizontally
            combined = np.hstack((frame, depth_color))
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('RGB Input | Depth Estimation', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()