"""
Comprehensive Accuracy & Confidence Benchmark Test
Evaluates Top-1 accuracy, crop identification, and confidence distribution.
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
    generate_corn_blight,
    generate_wheat_yellow_rust,
    generate_cotton_bacterial_blight,
    generate_banana_black_sigatoka,
    generate_coffee_leaf_rust
)
from backend.classifier import CropInferenceEngine

eng = CropInferenceEngine()

test_cases = [
    ("Wheat Yellow Rust", generate_wheat_yellow_rust(), "Wheat___Yellow_Rust", "Wheat"),
    ("Cotton Bacterial Blight", generate_cotton_bacterial_blight(), "Cotton___Bacterial_Blight", "Cotton"),
    ("Banana Black Sigatoka", generate_banana_black_sigatoka(), "Banana___Black_Sigatoka", "Banana / Plantain"),
    ("Coffee Leaf Rust", generate_coffee_leaf_rust(), "Coffee___Leaf_Rust", "Coffee"),
    ("Tomato Late Blight", generate_tomato_late_blight(), "Tomato___Late_blight", "Tomato"),
    ("Potato Early Blight", generate_potato_early_blight(), "Potato___Early_blight", "Potato"),
    ("Sugarcane Red Rot", generate_sugarcane_red_rot(), "Sugarcane___Red_Rot", "Sugarcane"),
    ("Sugarcane Rust", generate_sugarcane_rust(), "Sugarcane___Rust", "Sugarcane"),
    ("Apple Scab", generate_apple_scab(), "Apple___Apple_scab", "Apple"),
    ("Corn Blight", generate_corn_blight(), "Corn_(maize)___Northern_Leaf_Blight", "Corn (Maize)"),
    ("Squash Powdery Mildew", generate_squash_powdery_mildew(), "Squash___Powdery_mildew", "Squash / Cucurbit"),
    ("Healthy Tomato", generate_healthy_tomato(), "Tomato___healthy", "Tomato")
]

print("=" * 75)
print("BENCHMARKING ACCURACY ACROSS ALL SPECIMENS")
print("=" * 75)

all_passed = True
for name, img, expected_class, expected_crop in test_cases:
    res = eng.predict(img)
    top = res["top_prediction"]
    crop = res["crop_identification"]["detected_crop"]
    conf = top["confidence"]
    
    is_correct = (top["class_name"] == expected_class)
    if not is_correct: all_passed = False
    
    status = "[PASS]" if is_correct else "[FAIL]"
    print(f"{status} | {name:22s} -> Predicted: {top['class_name']:35s} ({conf:5.1f}%) | Crop: {crop}")


print("=" * 75)
print(f"Overall Benchmark: {'100% ACCURATE' if all_passed else 'NEEDS TUNING'}")
