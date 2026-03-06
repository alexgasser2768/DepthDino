import torch
from urllib.request import urlopen
from PIL import Image
import timm

# Load image
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# Create model
model = timm.create_model(
    'convnext_tiny.dinov3_lvd1689m',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# Get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Prepare input
dummy_input = transforms(img).unsqueeze(0)  # unsqueeze single image into batch of 1

# Run inference to get output shapes (optional, but good for verification)
with torch.no_grad():
    output = model(dummy_input)

print("Output shapes:")
for i, o in enumerate(output):
    print(f"Output {i}: {o.shape}")

# Export to ONNX
onnx_filename = "convnext_tiny_dinov3.onnx"
output_names = [f"feature_{i}" for i in range(len(output))]

print(f"\nExporting to {onnx_filename}...")
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            # You can uncomment these if you want dynamic resolution support
            # 'input': {0: 'batch_size', 2: 'height', 3: 'width'} 
        }
    )
    print("Successfully exported model to ONNX.")
except Exception as e:
    print(f"Error exporting to ONNX: {e}")
