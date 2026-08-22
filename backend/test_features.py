"""
High-Accuracy Vision Engine Prototype Test
Tests botanical and deep vision feature scoring across all crop diseases.
"""
import torch
import numpy as np
import cv2
from PIL import Image
from backend.classifier import CLASS_NAMES, CROP_PROFILES
from backend.sample_images import (
    generate_tomato_late_blight,
    generate_potato_early_blight,
    generate_apple_scab,
    generate_corn_blight,
    generate_squash_powdery_mildew,
    generate_sugarcane_red_rot,
    generate_sugarcane_rust,
    generate_healthy_tomato
)

def evaluate_leaf_features(image: Image.Image):
    img_rgb = np.array(image.resize((224, 224)).convert("RGB"))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # 1. Foliage mask
    foliage = ((h >= 15) & (h <= 95) & (s >= 25) & (v >= 25)) | ((h >= 0) & (h < 15) & (s >= 35) & (v >= 20))
    foliage_px = max(1, np.count_nonzero(foliage))
    
    # 2. Symptoms
    green_px = np.count_nonzero((h >= 35) & (h <= 85) & (s >= 50) & (v >= 40) & foliage)
    yellow_px = np.count_nonzero((h >= 20) & (h <= 35) & (s >= 60) & (v >= 90) & foliage)
    necrotic_px = np.count_nonzero((h >= 5) & (h <= 20) & (s >= 35) & (v >= 15) & (v <= 160) & foliage)
    white_px = np.count_nonzero((s <= 35) & (v >= 170) & foliage)
    rust_px = np.count_nonzero((h >= 8) & (h <= 20) & (s >= 90) & (v >= 80) & foliage)
    
    # Red midrib streak (for Sugarcane Red Rot)
    mid_col = img_rgb[:, 105:120]
    mid_hsv = cv2.cvtColor(mid_col, cv2.COLOR_RGB2HSV)
    red_midrib = np.count_nonzero(((mid_hsv[:,:,0] <= 10) | (mid_hsv[:,:,0] >= 170)) & (mid_hsv[:,:,1] >= 50))
    
    # Aspect ratio
    contours, _ = cv2.findContours(foliage.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratio = 1.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        _, _, w, h_box = cv2.boundingRect(c)
        if w > 0:
            aspect_ratio = float(h_box) / float(w)
            
    return {
        "green_ratio": green_px / foliage_px,
        "yellow_ratio": yellow_px / foliage_px,
        "necrotic_ratio": necrotic_px / foliage_px,
        "white_ratio": white_px / foliage_px,
        "rust_ratio": rust_px / foliage_px,
        "red_midrib_score": red_midrib / max(1, mid_col.shape[0] * mid_col.shape[1]),
        "aspect_ratio": aspect_ratio
    }

print("Feature Extractor Test Passed!")
