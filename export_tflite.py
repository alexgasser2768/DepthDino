import torch
import litert_torch
import litert_torch.quantize.quant_config
import PIL.Image
import torchvision.transforms as T
import glob
import os
import numpy as np
from model import ConvNeXtDepthModel

def export_to_pixel_tflite():
    # 1. Configuration
    weights_path = "weights/decoder/best_model_phone.pth"
    output_path = "weights/decoder/best_model_phone_tpu.tflite"
    input_size = (192, 192) 
    
    # 2. Load Model
    print(f"Loading weights from {weights_path}...")
    model = ConvNeXtDepthModel(
        arch='convnext_tiny.dinov3_lvd1689m', 
        mlp_weights_path=weights_path, 
        pretrained=False
    )
    model.eval()

    # 3. Representative Dataset for TPU Calibration
    def representative_dataset():
        image_paths = sorted(glob.glob("data/*.jpg"))
        if not image_paths:
            print("Warning: No data/ folder found. Using random data.")
            for _ in range(50):
                yield (torch.randn(1, 3, *input_size),)
            return

        print(f"Calibrating on {min(100, len(image_paths))} images for TPU...")
        transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for path in image_paths[:100]:
            try:
                img = PIL.Image.open(path).convert("RGB")
                yield (transform(img).unsqueeze(0),)
            except: continue

    # 4. Convert using LiteRT
    print("Converting to TFLite (LiteRT) for Pixel G2 TPU...")
    sample_input = torch.randn(1, 3, *input_size)
    
    # Updated API for LiteRT 0.8.0
    # PT2EQuantizer() is the modern class used for XNNPACK/TPU targets
    quantizer = litert_torch.quantize.PT2EQuantizer()
    
    quant_config = litert_torch.quantize.quant_config.QuantConfig(
        pt2e_quantizer=quantizer,
        representative_dataset=representative_dataset
    )

    # Convert the model
    edge_model = litert_torch.convert(
        model, 
        (sample_input,),
        quant_config=quant_config
    )

    # 5. Export and Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edge_model.export(output_path)
    print(f"\n--- SUCCESS ---")
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    export_to_pixel_tflite()
