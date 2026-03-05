```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
pip install -e .
```

```python
import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-small")
model = model.to(device=device)

# Run inference on images
images = ["image1.jpg", "image2.jpg"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
```