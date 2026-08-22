"""
Visual feature metric diagnostics
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.sample_images import (

    generate_tomato_late_blight,
    generate_potato_early_blight,
    generate_sugarcane_red_rot,
    generate_sugarcane_rust,
    generate_squash_powdery_mildew,
    generate_healthy_tomato,
    generate_apple_scab,
    generate_corn_blight
)
from backend.classifier import CropInferenceEngine

eng = CropInferenceEngine()

samples = [
    ('Tomato Late Blight', generate_tomato_late_blight(), 'Tomato___Late_blight'),
    ('Potato Early Blight', generate_potato_early_blight(), 'Potato___Early_blight'),
    ('Sugarcane Red Rot', generate_sugarcane_red_rot(), 'Sugarcane___Red_Rot'),
    ('Sugarcane Rust', generate_sugarcane_rust(), 'Sugarcane___Rust'),
    ('Apple Scab', generate_apple_scab(), 'Apple___Apple_scab'),
    ('Corn Blight', generate_corn_blight(), 'Corn_(maize)___Northern_Leaf_Blight'),
    ('Squash Mildew', generate_squash_powdery_mildew(), 'Squash___Powdery_mildew'),
    ('Healthy Tomato', generate_healthy_tomato(), 'Tomato___healthy')
]

for name, img, expected in samples:
    tensor, rgb = eng.preprocess_image(img)
    h = eng.analyze_visual_heuristics(rgb)
    print(f"{name:20s}: nec={h['necrotic_score']:.3f}, chl={h['chlorosis_score']:.3f}, pwd={h['powder_score']:.3f}, rst={h['rust_score']:.3f}, rmb={h['red_midrib_score']:.3f}, ar={h['aspect_ratio']:.2f}")
