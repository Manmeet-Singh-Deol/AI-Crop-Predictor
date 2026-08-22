"""
OpenCV-Based Crop Disease Severity Quantification & Leaf Area Segmentation
Calculates precise lesion percentage, infection classification stage, and visual diagnostic masks.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any

def quantify_severity(image: Image.Image) -> Dict[str, Any]:
    """
    Segment leaf blade and diseased lesions to calculate quantitative severity metrics.
    Returns infection percentage, severity stage, contour counts, and annotated visual mask.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_rgb = np.array(image.resize((400, 400)), dtype=np.uint8)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    # 1. Segment Entire Leaf Blade (excluding white/black/neutral backgrounds)
    # Leaf includes green, yellow, brown, and necrotic portions
    leaf_mask_hsv = (
        ((h >= 18) & (h <= 95) & (s >= 25) & (v >= 25)) |       # Greens & Yellows
        ((h >= 5) & (h < 25) & (s >= 35) & (v >= 25) & (v <= 230)) | # Brown/Orange spots
        ((s >= 40) & (v >= 30) & (v <= 220))                    # General saturated foliage
    ).astype(np.uint8) * 255
    
    # Morphological cleaning of leaf mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(leaf_mask_hsv, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    
    total_leaf_pixels = int(np.count_nonzero(leaf_mask))
    total_image_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    
    # Handle edge case where background segmentation fails
    if total_leaf_pixels < 2000:
        total_leaf_pixels = total_image_pixels
        leaf_mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8) * 255
        
    # 2. Segment Healthy Green Tissue
    healthy_green_mask = (
        (h >= 35) & (h <= 85) & (s >= 45) & (v >= 40) & (leaf_mask > 0)
    ).astype(np.uint8) * 255
    
    # 3. Segment Diseased / Damaged Lesions
    # A. Necrotic dark brown / black / dry lesions
    necrotic_mask = (
        (((h >= 0) & (h < 22)) | (h >= 160)) & 
        (s >= 30) & 
        (v >= 15) & (v <= 170) & 
        (leaf_mask > 0)
    ).astype(np.uint8) * 255
    
    # B. Chlorotic Yellowing / Yellow Halos
    chlorosis_mask = (
        (h >= 20) & (h <= 34) & (s >= 65) & (v >= 70) & (leaf_mask > 0)
    ).astype(np.uint8) * 255
    
    # C. Powdery Mildew White / Gray Coatings
    powder_mask = (
        (s <= 40) & (v >= 160) & (leaf_mask > 0)
    ).astype(np.uint8) * 255
    
    # Combined lesion mask
    lesion_mask_raw = cv2.bitwise_or(necrotic_mask, chlorosis_mask)
    lesion_mask_raw = cv2.bitwise_or(lesion_mask_raw, powder_mask)
    lesion_mask = cv2.morphologyEx(lesion_mask_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    infected_pixels = int(np.count_nonzero(lesion_mask))
    healthy_pixels = max(0, total_leaf_pixels - infected_pixels)
    
    # Calculate Infection Ratio (%)
    infection_pct = (infected_pixels / float(total_leaf_pixels)) * 100.0
    infection_pct = min(100.0, max(0.0, round(infection_pct, 2)))
    
    # Categorize Severity Stage
    if infection_pct < 1.5:
        severity_stage = "Healthy"
        stage_color = "#10b981"  # Emerald green
        urgency = "Normal"
        recommendation = "No active chemical or biological intervention required. Maintain standard crop nutrition and moisture management."
    elif infection_pct < 15.0:
        severity_stage = "Mild Infection"
        stage_color = "#f59e0b"  # Amber
        urgency = "Moderate"
        recommendation = "Early-stage pathogen onset detected. Apply biological controls (Neem oil, Bacillus subtilis, Trichoderma) and prune heavily affected lower leaves."
    elif infection_pct <= 40.0:
        severity_stage = "Moderate Infection"
        stage_color = "#f97316"  # Orange
        urgency = "High"
        recommendation = "Active disease progression across canopy. Initiate targeted systemic or protectant fungicide/bactericide spray protocol and improve aeration."
    else:
        severity_stage = "Severe / Critical Infection"
        stage_color = "#ef4444"  # Red
        urgency = "Critical"
        recommendation = "Extensive tissue necrosis threatening total crop loss. Apply immediate curative systemic fungicide/bactericide, isolate affected rows, and destroy infected residues."

    # 4. Generate Visual Overlay Mask
    # Create an RGB overlay highlighting lesions in Crimson Red and healthy leaf boundary in Cyan
    overlay_rgb = img_rgb.copy()
    
    # Draw transparent crimson tint over lesions
    red_tint = np.zeros_like(img_rgb)
    red_tint[:, :] = [239, 68, 68]  # Bright red
    
    mask_bool = lesion_mask > 0
    alpha = 0.55
    overlay_rgb[mask_bool] = (alpha * red_tint[mask_bool] + (1 - alpha) * overlay_rgb[mask_bool]).astype(np.uint8)
    
    # Find lesion contours and draw sharp outlines
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, -1, (255, 235, 59), 1)  # Yellow contour border
    
    # Draw leaf perimeter in Cyan
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, leaf_contours, -1, (6, 182, 212), 2)  # Cyan boundary
    
    # Encode overlay to base64 URL
    pil_overlay = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_overlay.save(buf, format="JPEG", quality=90)
    overlay_base64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    
    return {
        "severity_percentage": infection_pct,
        "severity_stage": severity_stage,
        "stage_color": stage_color,
        "urgency": urgency,
        "recommendation": recommendation,
        "leaf_area_pixels": total_leaf_pixels,
        "lesion_area_pixels": infected_pixels,
        "healthy_area_pixels": healthy_pixels,
        "lesion_count": len(contours),
        "severity_mask_image": overlay_base64
    }
