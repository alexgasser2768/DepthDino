import numpy as np
import matplotlib.pyplot as plt
import cv2, time, logging, argparse, os
import PIL.Image

# Optional imports based on engine selection
try:
    import torch
    from torchvision import transforms as T
    from model import ConvNeXtDepthModel
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Configuration
ONNX_MODEL_PATH = "weights/decoder/best_model_phone_fp16.onnx"
PYTORCH_WEIGHTS = "weights/decoder/best_model_phone_v2.pth"
CMAP = plt.get_cmap('magma')

logger = logging.getLogger(__name__) 
logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def preprocess_numpy(frame, size=(224, 224), dtype=np.float32):
    """Preprocess using only Numpy (for ONNX)"""
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(dtype) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=dtype)
    std = np.array([0.229, 0.224, 0.225], dtype=dtype)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)

def run_webcam(engine='onnx'):
    # --- 1. Model Initialization ---
    if engine == 'onnx':
        if ort is None: raise ImportError("onnxruntime not installed.")
        logger.info(f"Loading ONNX model: {ONNX_MODEL_PATH}")
        
        # Explicitly check for file existence
        if not os.path.exists(ONNX_MODEL_PATH):
            logger.error(f"ONNX model file not found: {ONNX_MODEL_PATH}")
            return

        # Try CUDA, then CPU
        try:
            session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            logger.info(f"ONNX Session initialized with providers: {session.get_providers()}")
        except Exception as e:
            logger.warning(f"CUDA initialization failed, falling back to CPU: {e}")
            session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        
        input_size = (224, 224)
    else:
        if torch is None: raise ImportError("torch/torchvision or model.py not found.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading PyTorch model on {device}: {PYTORCH_WEIGHTS}")
        model = ConvNeXtDepthModel(arch='convnext_tiny.dinov3_lvd1689m', mlp_weights_path=PYTORCH_WEIGHTS)
        model.to(device).eval()
        input_size = (640, 480) # PyTorch version uses higher res by default

    # --- 2. Webcam Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open webcam.")
        return

    cv2.namedWindow('Depth Estimation', cv2.WINDOW_NORMAL)
    cv2.startWindowThread() # Helps with some Linux window managers
    print(f"Running {engine.upper()} engine. Press 'q' to quit.")
    
    first_frame = True
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # --- 3. Inference ---
        if engine == 'onnx':
            input_data = preprocess_numpy(frame, size=input_size, dtype=np.float16)
            outputs = session.run(['output'], {'input': input_data})
            depth_np = outputs[0].squeeze().astype(np.float32)
        else:
            with torch.no_grad():
                img_rgb = cv2.cvtColor(cv2.resize(frame, input_size), cv2.COLOR_BGR2RGB)
                img_pil = PIL.Image.fromarray(img_rgb)
                input_tensor = model.transforms(img_pil).unsqueeze(0).to(device)
                depth_map = model(input_tensor)
                depth_np = depth_map.squeeze().cpu().numpy()

        # --- 4. Post-processing & Visualization ---
        depth_np = np.nan_to_num(depth_np)

        d_min, d_max = depth_np.min(), depth_np.max()
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-5)
        
        depth_color = (255 * CMAP(depth_norm)[:, :, :3]).astype(np.uint8)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
        depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

        combined = np.hstack((frame, depth_color))
        fps = 1.0 / (time.time() - start_time)
        
        # Display the actual provider being used
        active_provider = session.get_provider_options().keys() if engine == 'onnx' else ['PyTorch']
        provider_name = list(active_provider)[0]
        print(f"\rEngine: {engine} | Provider: {provider_name} | FPS: {fps:.1f} | Frame: {frame.shape}", end="")
        
        # Save a debug frame on the first successful loop
        if first_frame:
            cv2.imwrite("debug_frame.jpg", combined)
            logger.info("Saved debug_frame.jpg to disk.")
            first_frame = False

        cv2.putText(combined, f"FPS: {fps:.1f} ({engine.upper()})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Depth Estimation', combined)

        # Force a small wait and process UI events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DinoDepth Webcam Demo")
    parser.add_argument("--engine", type=str, choices=['pytorch', 'onnx'], default='onnx', 
                        help="Inference engine to use (default: onnx)")
    args = parser.parse_args()
    
    run_webcam(engine=args.engine)
