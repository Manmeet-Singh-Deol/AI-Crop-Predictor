"""
Cross-Confusion Robustness Test
Tests that sugarcane and tomato are correctly distinguished even when
the leaf is cropped, rotated, or photographed at unusual angles.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from backend.classifier import CropInferenceEngine

eng = CropInferenceEngine()

def make_sugarcane_square():
    """Sugarcane leaf cropped to a SQUARE frame (AR ~ 1.0) - tests non-AR features."""
    img = Image.new("RGB", (400, 400), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # Wide sugarcane blade filling entire frame (AR goes away)
    draw.polygon([(20, 20), (380, 20), (380, 380), (20, 380)], fill=(40, 130, 50))
    # White midrib
    draw.line([(200, 20), (200, 380)], fill=(225, 240, 220), width=14)
    # Red rot lesion on midrib
    draw.rectangle([195, 80, 205, 320], fill=(165, 20, 25))
    draw.rectangle([197, 120, 203, 280], fill=(210, 185, 130))
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))

def make_tomato_tall():
    """Tomato leaf in a TALL frame (AR ~ 2.0) - tests non-AR features."""
    img = Image.new("RGB", (250, 500), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    # Serrated compound tomato leaf shape 
    cx, cy = 125, 250
    pts = []
    for i in range(60):
        angle = (2 * np.pi * i) / 60
        r = 100 * (1.0 + 0.25 * np.cos(angle) - 0.15 * np.sin(3 * angle))
        x = cx + r * np.sin(angle) + np.random.uniform(-5, 5)
        y = cy - r * np.cos(angle) * 1.3 + np.random.uniform(-5, 5)
        pts.append((x, y))
    draw.polygon(pts, fill=(46, 125, 50))
    # Veins
    for dy in range(-120, 120, 30):
        draw.line([(cx, cy + dy), (cx - 60, cy + dy - 20)], fill=(55, 140, 60), width=2)
        draw.line([(cx, cy + dy), (cx + 60, cy + dy - 20)], fill=(55, 140, 60), width=2)
    # Late blight lesions
    draw.ellipse([70, 180, 180, 290], fill=(190, 180, 40))
    draw.ellipse([85, 200, 165, 270], fill=(55, 38, 25))
    draw.ellipse([90, 300, 160, 370], fill=(45, 30, 20))
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))

print("=" * 70)
print("CROSS-CONFUSION ROBUSTNESS TEST")
print("=" * 70)

# Test 1: Square-cropped sugarcane (should NOT be confused with tomato)
res1 = eng.predict(make_sugarcane_square())
t1 = res1["top_prediction"]
crop1 = res1["crop_identification"]["detected_crop"]
h1 = res1["heuristics"]
ok1 = "Sugarcane" in t1["crop"]
print(f"{'[PASS]' if ok1 else '[FAIL]'} | Square Sugarcane    -> {t1['class_name']:35s} ({t1['confidence']:5.1f}%) | Crop: {crop1}")
print(f"       Features: pvr={h1.get('parallel_vein_ratio',0):.3f}, cu={h1.get('color_uniformity',0):.3f}, ar={h1['aspect_ratio']:.2f}, rmb={h1['red_midrib_score']:.3f}")

# Test 2: Tall-framed tomato (should NOT be confused with sugarcane)
res2 = eng.predict(make_tomato_tall())
t2 = res2["top_prediction"]
crop2 = res2["crop_identification"]["detected_crop"]
h2 = res2["heuristics"]
ok2 = "Tomato" in t2["crop"]
print(f"{'[PASS]' if ok2 else '[FAIL]'} | Tall Tomato         -> {t2['class_name']:35s} ({t2['confidence']:5.1f}%) | Crop: {crop2}")
print(f"       Features: pvr={h2.get('parallel_vein_ratio',0):.3f}, cu={h2.get('color_uniformity',0):.3f}, ar={h2['aspect_ratio']:.2f}, rmb={h2['red_midrib_score']:.3f}")

print("=" * 70)
both_pass = ok1 and ok2
print(f"Cross-Confusion Test: {'ROBUST - NO CONFUSION' if both_pass else 'NEEDS WORK'}")
