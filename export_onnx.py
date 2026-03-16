import torch
import onnx
from onnxconverter_common import float16
from model import ConvNeXtDepthModel

def export_onnx_fp16():
    weights_path = "weights/decoder/best_model_phone.pth"
    fp32_onnx_path = "weights/decoder/best_model_phone.onnx"
    fp16_onnx_path = "weights/decoder/best_model_phone_fp16.onnx"
    input_size = (224, 224)

    # 1. Export to FP32 ONNX
    print(f"Loading weights from {weights_path}...")
    model = ConvNeXtDepthModel(
        arch='convnext_tiny.dinov3_lvd1689m', 
        mlp_weights_path=weights_path, 
        pretrained=False
    )
    model.eval()

    dummy_input = torch.randn(1, 3, *input_size)
    print(f"Ensuring FP32 ONNX exists: {fp32_onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=17, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 2. Convert to Float16
    # This keeps high precision for depth estimation but cuts memory/bandwidth in half
    print("Converting to ONNX Float16 (High Accuracy)...")
    model_fp32 = onnx.load(fp32_onnx_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    
    onnx.save(model_fp16, fp16_onnx_path)
    print(f"Successfully saved FLOAT16 ONNX model to: {fp16_onnx_path}")

if __name__ == "__main__":
    export_onnx_fp16()
