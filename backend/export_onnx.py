"""
Model Exporter for Edge & Mobile Deployment (TorchScript & ONNX)
Converts PyTorch MobileNetV3 crop disease classification model into optimized serialized formats.
"""

import os
import sys
import torch
from backend.classifier import get_inference_engine, IMAGE_SIZE, CLASS_NAMES

def export_model_to_torchscript(output_path: str = "crop_disease_model.pt") -> str:
    """Export model to optimized TorchScript format for mobile/edge embedded deployment."""
    engine = get_inference_engine()
    model = engine.model
    model.eval()
    
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(engine.device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Model Export] Successfully exported TorchScript model to {output_path} ({file_size_mb:.2f} MB)")
    return output_path

def export_model_to_onnx(output_path: str = "crop_disease_model.onnx") -> str:
    """Export model to ONNX if onnxscript is available, or TorchScript fallback."""
    try:
        import onnx
        import onnxscript
        engine = get_inference_engine()
        model = engine.model
        model.eval()
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(engine.device)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["class_probabilities"]
        )
        return output_path
    except (ImportError, Exception) as e:
        print(f"[Model Export] ONNX library not found ({e}). Exporting TorchScript model instead.")
        ts_path = output_path.replace(".onnx", ".pt")
        return export_model_to_torchscript(ts_path)

if __name__ == "__main__":
    export_model_to_torchscript()
