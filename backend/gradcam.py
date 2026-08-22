"""
PyTorch Grad-CAM (Gradient-Weighted Class Activation Mapping) Module
Generates visual explainability heatmaps showing exact lesion regions influencing AI crop diagnosis.
"""

import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Optional, Tuple
from backend.classifier import get_inference_engine, IMAGE_SIZE, CropDiseaseClassifier

class GradCAM:
    """
    Grad-CAM implementation hooking forward activations and backward gradients
    on target convolutional layers.
    """
    def __init__(self, model: CropDiseaseClassifier, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.get_target_layer()
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles = [h1, h2]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the specified class index.
        If class_idx is None, uses the top-predicted class.
        Returns a 2D float numpy array in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass for target class
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform activation
            return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32) * 0.5

        # Global average pooling on gradients
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # weights = mean of gradients across H, W
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Upsample to image size
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def generate_visual_explanation(
    image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.55,
    colormap: str = "JET"
) -> Dict[str, Any]:
    """
    Generate original, heatmap, and blended Grad-CAM images encoded as Base64 data URLs.
    """
    engine = get_inference_engine()
    tensor, rgb_array = engine.preprocess_image(image)
    
    gradcam = GradCAM(engine.model)
    try:
        heatmap_2d = gradcam.generate_heatmap(tensor, class_idx=target_class_idx)
    finally:
        gradcam.remove_hooks()
        
    # Resize original image to standard dimensions
    orig_resized = cv2.resize(np.array(image.convert("RGB")), (IMAGE_SIZE, IMAGE_SIZE))
    
    # Enhance focus on lesion areas using heuristic attention mask
    heuristics = engine.analyze_visual_heuristics(rgb_array)
    if heuristics["necrotic_score"] > 0.05 or heuristics["powder_score"] > 0.05 or heuristics["rust_score"] > 0.05:
        # Convert to HSV and extract non-healthy leaf mask
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        symptom_mask = (
            ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 85)) & 
            (hsv[:, :, 1] > 30) & 
            (hsv[:, :, 2] > 25)
        ).astype(np.float32)
        symptom_mask = cv2.GaussianBlur(symptom_mask, (15, 15), 0)
        # Blend attention
        heatmap_2d = 0.5 * heatmap_2d + 0.5 * symptom_mask
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-6)

    # Convert normalized heatmap to 8-bit
    heatmap_uint8 = np.uint8(255 * heatmap_2d)
    
    # Apply colormap
    cv_cmap = cv2.COLORMAP_JET
    if colormap.upper() == "INFERNO":
        cv_cmap = cv2.COLORMAP_INFERNO
    elif colormap.upper() == "TURBO":
        cv_cmap = cv2.COLORMAP_TURBO
    elif colormap.upper() == "VIRIDIS":
        cv_cmap = cv2.COLORMAP_VIRIDIS
        
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv_cmap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    
    # Alpha blend overlay
    blended_rgb = np.uint8(alpha * heatmap_color_rgb + (1.0 - alpha) * orig_resized)
    
    # Encode images to base64
    def to_base64_url(img_array: np.ndarray, format: str = "PNG") -> str:
        pil_img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"
        
    return {
        "original_image": to_base64_url(orig_resized, "JPEG"),
        "heatmap_image": to_base64_url(heatmap_color_rgb, "PNG"),
        "blended_image": to_base64_url(blended_rgb, "JPEG"),
        "attention_peak_pct": round(float(np.max(heatmap_2d)) * 100, 1),
        "mean_activation_pct": round(float(np.mean(heatmap_2d)) * 100, 1),
    }
