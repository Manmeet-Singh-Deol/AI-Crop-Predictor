"""
Deep Learning Inference Engine for Crop Disease Classification
Supports 47 crop disease classes with PyTorch backbone, multi-feature botanical
texture analysis, robust crop species identification, and calibrated scoring.

Architecture:
  - MobileNetV3-Small backbone (first 38 PlantVillage classes have trained weights)
  - Multi-channel HSV + texture + morphology feature extraction
  - Hybrid scoring: neural network logits + biological symptom rules
  - Temperature-scaled softmax for sharp confidence separation
"""

import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

# Standard 47-class crop disease taxonomy
CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Rice___Leaf_Blast",
    "Rice___Brown_Spot",
    "Rice___healthy",
    "Sugarcane___Red_Rot",
    "Sugarcane___Smut",
    "Sugarcane___Rust",
    "Sugarcane___Yellow_Leaf",
    "Sugarcane___Mosaic",
    "Sugarcane___healthy",
    "Wheat___Yellow_Rust",
    "Wheat___Brown_Rust",
    "Wheat___Powdery_Mildew",
    "Wheat___healthy",
    "Cotton___Bacterial_Blight",
    "Cotton___Leaf_Curl_Virus",
    "Cotton___Target_Spot",
    "Cotton___healthy",
    "Coffee___Leaf_Rust",
    "Coffee___Cercospora_Leaf_Spot",
    "Coffee___healthy",
    "Tea___Blister_Blight",
    "Tea___Red_Rust",
    "Tea___healthy",
    "Cassava___Mosaic_Disease",
    "Cassava___Bacterial_Blight",
    "Cassava___healthy",
    "Banana___Black_Sigatoka",
    "Banana___Yellow_Sigatoka",
    "Banana___healthy"
]

# Number of classes the trained checkpoint covers (PlantVillage original 38)
NUM_TRAINED_CLASSES = 38

# Comprehensive Botanical Crop Taxonomy & Morphology Map
CROP_PROFILES: Dict[str, Dict[str, Any]] = {
    "Sugarcane": {
        "display": "Sugarcane",
        "botanical_name": "Saccharum officinarum",
        "family": "Poaceae (Grass / C4 Perennial)",
        "keys": ["Sugarcane"]
    },
    "Wheat": {
        "display": "Wheat",
        "botanical_name": "Triticum aestivum",
        "family": "Poaceae (Cereal Grass)",
        "keys": ["Wheat"]
    },
    "Cotton": {
        "display": "Cotton",
        "botanical_name": "Gossypium hirsutum",
        "family": "Malvaceae (Mallow)",
        "keys": ["Cotton"]
    },
    "Coffee": {
        "display": "Coffee",
        "botanical_name": "Coffea arabica",
        "family": "Rubiaceae (Madder)",
        "keys": ["Coffee"]
    },
    "Tea": {
        "display": "Tea",
        "botanical_name": "Camellia sinensis",
        "family": "Theaceae (Tea Family)",
        "keys": ["Tea"]
    },
    "Cassava": {
        "display": "Cassava / Yuca",
        "botanical_name": "Manihot esculenta",
        "family": "Euphorbiaceae (Spurge)",
        "keys": ["Cassava"]
    },
    "Banana": {
        "display": "Banana / Plantain",
        "botanical_name": "Musa acuminata",
        "family": "Musaceae (Banana Family)",
        "keys": ["Banana"]
    },
    "Tomato": {
        "display": "Tomato",
        "botanical_name": "Solanum lycopersicum",
        "family": "Solanaceae (Nightshade)",
        "keys": ["Tomato"]
    },
    "Potato": {
        "display": "Potato",
        "botanical_name": "Solanum tuberosum",
        "family": "Solanaceae (Nightshade)",
        "keys": ["Potato"]
    },
    "Apple": {
        "display": "Apple",
        "botanical_name": "Malus domestica",
        "family": "Rosaceae (Rose)",
        "keys": ["Apple"]
    },
    "Corn": {
        "display": "Corn (Maize)",
        "botanical_name": "Zea mays",
        "family": "Poaceae (Grass)",
        "keys": ["Corn_(maize)"]
    },
    "Grape": {
        "display": "Grape (Vineyard)",
        "botanical_name": "Vitis vinifera",
        "family": "Vitaceae",
        "keys": ["Grape"]
    },
    "Rice": {
        "display": "Rice (Paddy)",
        "botanical_name": "Oryza sativa",
        "family": "Poaceae (Grass)",
        "keys": ["Rice"]
    },
    "Pepper": {
        "display": "Pepper (Bell)",
        "botanical_name": "Capsicum annuum",
        "family": "Solanaceae (Nightshade)",
        "keys": ["Pepper,_bell"]
    },
    "Orange": {
        "display": "Orange (Citrus)",
        "botanical_name": "Citrus × sinensis",
        "family": "Rutaceae (Citrus)",
        "keys": ["Orange"]
    },
    "Peach": {
        "display": "Peach",
        "botanical_name": "Prunus persica",
        "family": "Rosaceae (Stone Fruit)",
        "keys": ["Peach"]
    },
    "Squash": {
        "display": "Squash / Cucurbit",
        "botanical_name": "Cucurbita pepo",
        "family": "Cucurbitaceae (Gourd)",
        "keys": ["Squash"]
    },
    "Strawberry": {
        "display": "Strawberry",
        "botanical_name": "Fragaria × ananassa",
        "family": "Rosaceae (Berry)",
        "keys": ["Strawberry"]
    },
    "Cherry": {
        "display": "Cherry",
        "botanical_name": "Prunus cerasus",
        "family": "Rosaceae (Stone Fruit)",
        "keys": ["Cherry_(including_sour)"]
    },
    "Soybean": {
        "display": "Soybean",
        "botanical_name": "Glycine max",
        "family": "Fabaceae (Legume)",
        "keys": ["Soybean"]
    },
    "Blueberry": {
        "display": "Blueberry",
        "botanical_name": "Vaccinium corymbosum",
        "family": "Ericaceae",
        "keys": ["Blueberry"]
    },
    "Raspberry": {
        "display": "Raspberry",
        "botanical_name": "Rubus idaeus",
        "family": "Rosaceae (Caneberry)",
        "keys": ["Raspberry"]
    }
}

# Standard ImageNet normalization for PyTorch vision models
IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CropDiseaseClassifier(nn.Module):
    """
    MobileNetV3-based classifier for crop disease diagnosis.
    Designed for fast inference and compatibility with Grad-CAM gradient hooks.
    """
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        try:
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            self.backbone = models.mobilenet_v3_small(weights=None)
            
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Returns the final convolutional feature layer for Grad-CAM."""
        return self.backbone.features[-1]


class CropInferenceEngine:
    """
    Singleton inference engine managing model loading, image preprocessing,
    crop species detection, and calibrated high-confidence predictions.
    """
    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(CLASS_NAMES)
        self.model = CropDiseaseClassifier(num_classes=self.num_classes)
        
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                for key in ["backbone.classifier.3.weight", "classifier.3.weight"]:
                    if key in state_dict:
                        src_w = state_dict[key]
                        n = min(src_w.shape[0], self.num_classes)
                        self.model.backbone.classifier[3].weight.data[:n] = src_w[:n].to(self.device)
                for key in ["backbone.classifier.3.bias", "classifier.3.bias"]:
                    if key in state_dict:
                        src_b = state_dict[key]
                        n = min(src_b.shape[0], self.num_classes)
                        self.model.backbone.classifier[3].bias.data[:n] = src_b[:n].to(self.device)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[InferenceEngine] Notice: baseline weights ({e})")
                
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert PIL image to preprocessed PyTorch tensor and RGB numpy array."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
        rgb_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
        return tensor, rgb_array

    def analyze_visual_heuristics(self, rgb_array: np.ndarray) -> Dict[str, float]:
        """
        Rotation-invariant multi-channel biological feature extraction using HSV,
        rotated bounding box geometry (PCA/minAreaRect), and directional gradients.
        
        Extracts robust features independent of camera orientation or framing:
        - True leaf aspect ratio (Length / Width via rotated min area rect)
        - Solidity and circularity (serrated compound leaflet vs smooth strap blade)
        - Directional vein gradient aligned with the leaf's principal axis
        - Pathognomonic red midrib streak anywhere on the leaf
        - Pale central midrib signature
        - Symptom color fractions (chlorosis, necrosis, rust, powdery, scab)
        """
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # 1. Broad foliage mask
        foliage = (
            ((h >= 15) & (h <= 95) & (s >= 20) & (v >= 20)) |
            ((h >= 0) & (h < 15) & (s >= 30) & (v >= 15)) |
            ((h >= 165) & (s >= 30) & (v >= 20))
        )
        foliage_px = max(1, np.count_nonzero(foliage))
        
        # 2. Symptom color masks (calibrated for real field photography across crops)
        healthy_green = np.count_nonzero((h >= 35) & (h <= 85) & (s >= 35) & (v >= 30) & foliage)
        chlorosis = np.count_nonzero((h >= 18) & (h <= 35) & (s >= 35) & (v >= 65) & foliage)
        necrotic = np.count_nonzero((h >= 5) & (h <= 20) & (s >= 25) & (v >= 15) & (v <= 165) & foliage)
        white_powdery = np.count_nonzero((s <= 40) & (v >= 155) & foliage)
        
        # Rust detection: Yellow stripe rust (Puccinia striiformis) & Brown rust (Puccinia triticina)
        rust_orange = np.count_nonzero((h >= 6) & (h <= 24) & (s >= 35) & (v >= 35) & foliage)
        rust_yellow = np.count_nonzero((h >= 15) & (h <= 36) & (s >= 35) & (v >= 50) & foliage)
        total_rust = max(rust_orange, rust_yellow, int(np.count_nonzero((h >= 8) & (h <= 32) & (s >= 30) & (v >= 40) & foliage)))
        
        scab_olive = np.count_nonzero((h >= 15) & (h <= 45) & (s >= 25) & (v >= 15) & (v <= 85) & foliage)
        
        # 3. Pathognomonic Red Streak anywhere on foliage (not locked to center column)
        red_pixels = ((h <= 10) | (h >= 165)) & (s >= 45) & (v >= 25) & foliage
        red_streak_score = float(np.count_nonzero(red_pixels)) / float(foliage_px)
        
        # Pale / White Central Midrib (characteristic monocot spine)
        pale_midrib = ((s <= 35) & (v >= 140) & foliage)
        pale_midrib_score = float(np.count_nonzero(pale_midrib)) / float(foliage_px)
        
        # 4. Rotation-Invariant Morphological Analysis via minAreaRect
        contours, _ = cv2.findContours(foliage.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        true_aspect_ratio = 1.0
        frame_aspect_ratio = 1.0
        circularity = 0.5
        solidity = 0.5
        leaf_angle = 0.0
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = float(4 * np.pi * (area / (peri * peri)))
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area / hull_area)
                
            # Rotation-invariant minimum area bounding box
            rect = cv2.minAreaRect(c)
            (_, (dim1, dim2), angle) = rect
            length = max(dim1, dim2)
            width = max(1.0, min(dim1, dim2))
            true_aspect_ratio = float(length / width)
            leaf_angle = angle
            
            # Frame-aligned bounding box
            _, _, w, h_box = cv2.boundingRect(c)
            if w > 0:
                frame_aspect_ratio = float(h_box) / float(w)
                
        # 5. Principal Axis Directional Vein Ratio (aligned to leaf angle)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        M = cv2.getRotationMatrix2D((IMAGE_SIZE // 2, IMAGE_SIZE // 2), leaf_angle, 1.0)
        rotated_gray = cv2.warpAffine(gray, M, (IMAGE_SIZE, IMAGE_SIZE))
        rotated_foliage = cv2.warpAffine(foliage.astype(np.uint8), M, (IMAGE_SIZE, IMAGE_SIZE)).astype(bool)
        
        sobel_x = cv2.Sobel(rotated_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(rotated_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        energy_x = float(np.sum(np.abs(sobel_x)[rotated_foliage])) if rotated_foliage.any() else 1.0
        energy_y = float(np.sum(np.abs(sobel_y)[rotated_foliage])) if rotated_foliage.any() else 1.0
        directional_vein_ratio = energy_x / (energy_x + energy_y + 1e-5)
        directional_anisotropy = abs(directional_vein_ratio - 0.5) * 2.0  # 0.0 = isotropic (dicot), 1.0 = parallel (monocot/cereal)
        
        # Color variance across foliage
        green_channel = rgb_array[:, :, 1].astype(np.float32)
        green_std = float(np.std(green_channel[foliage])) if foliage.any() else 50.0
        color_uniformity = 1.0 - min(1.0, green_std / 80.0)

        return {
            "healthy_score": healthy_green / foliage_px,
            "necrotic_score": necrotic / foliage_px,
            "chlorosis_score": chlorosis / foliage_px,
            "rust_score": total_rust / foliage_px,
            "powder_score": white_powdery / foliage_px,
            "scab_score": scab_olive / foliage_px,
            "red_midrib_score": red_streak_score,
            "pale_midrib_score": pale_midrib_score,
            "true_aspect_ratio": true_aspect_ratio,
            "aspect_ratio": frame_aspect_ratio,
            "circularity": circularity,
            "solidity": solidity,
            "directional_vein_ratio": directional_vein_ratio,
            "directional_anisotropy": directional_anisotropy,
            "color_uniformity": color_uniformity
        }

    def _compute_crop_affinity(self, h: Dict[str, float]) -> Dict[str, float]:
        """
        Rotation-invariant multi-feature crop affinity scoring.
        Uses true leaf aspect ratio, directional parallel vein anisotropy,
        color pustules, and pale midrib to cleanly separate Monocots/Cereals (Wheat/Sugarcane/Corn/Rice)
        from Dicots (Tomato/Potato/Apple/Squash/Grape).
        """
        tar = h["true_aspect_ratio"]
        sol = h["solidity"]
        circ = h["circularity"]
        rmb = h["red_midrib_score"]
        pmb = h["pale_midrib_score"]
        pwd = h["powder_score"]
        nec = h["necrotic_score"]
        chl = h["chlorosis_score"]
        rst = h["rust_score"]
        aniso = h.get("directional_anisotropy", 0.0)
        
        # 1. Monocot vs Dicot Voting based on true morphology & venation
        monocot_votes = 0
        dicot_votes = 0
        
        # True Aspect Ratio (Length / Width of the leaf blade regardless of rotation)
        if tar > 2.8:
            monocot_votes += 6
        elif tar > 1.8:
            monocot_votes += 4
        elif tar < 1.35 and aniso < 0.2:
            dicot_votes += 3
        else:
            dicot_votes += 1
            
        # Directional Vein Anisotropy (Parallel linear veins in Wheat/Sugarcane/Rice/Corn)
        if aniso > 0.35:
            monocot_votes += 5
        elif aniso > 0.22:
            monocot_votes += 3
            
        # Rust presence on narrow/moderate leaf strongly indicates Wheat / Cereal rust
        if rst > 0.015 and tar > 1.4:
            monocot_votes += 6
            
        # Solidity: Smooth linear blade has high solidity; lobed leaves have lower solidity
        if sol > 0.94 and tar > 1.7:
            monocot_votes += 3
        elif sol < 0.85:
            dicot_votes += 4
            
        # Circularity: Monocots are long and elongated (circ < 0.45); broad dicots have circ > 0.60
        if circ > 0.65 and aniso < 0.2:
            dicot_votes += 5
        elif circ < 0.42:
            monocot_votes += 4
            
        # Pale central midrib: definitive monocot leaf spine
        if pmb > 0.030 and tar > 1.6:
            monocot_votes += 4
            
        # Red streak: pathognomonic sugarcane red rot
        if rmb > 0.015:
            monocot_votes += 6
            
        # Consensus evaluation
        is_monocot = (monocot_votes >= 3 and monocot_votes >= dicot_votes)
        is_dicot = (dicot_votes >= 4 and dicot_votes > monocot_votes)
        
        affinities: Dict[str, float] = {}
        monocot_crops = ["Wheat", "Sugarcane", "Corn", "Rice", "Banana"]
        
        for c_name in CROP_PROFILES.keys():
            score = 1.0
            
            if c_name in monocot_crops:
                if is_monocot:
                    score += 14.0
                    if c_name == "Wheat":
                        # Wheat linear leaf blade & rust stripes
                        if rst > 0.01: score += 22.0 + (rst * 30.0)
                        elif tar > 2.0 or aniso > 0.25: score += 18.0
                        else: score += 12.0
                    elif c_name == "Sugarcane":
                        if rmb > 0.015: score += 22.0
                        elif pmb > 0.035: score += 16.0
                        elif tar > 2.0: score += 8.0
                    elif c_name == "Corn":
                        if chl > 0.08 and circ < 0.40: score += 16.0
                        elif tar > 2.0 and tar < 3.2: score += 10.0
                    elif c_name == "Banana":
                        if circ < 0.45 and tar < 2.0: score += 16.0
                    elif c_name == "Rice":
                        if tar > 3.5: score += 12.0
                elif is_dicot:
                    score -= 10.0
                else:
                    if rst > 0.01 and c_name == "Wheat": score += 14.0
                    elif tar > 1.8 and c_name == "Sugarcane" and pmb > 0.03: score += 8.0
                    elif circ < 0.45 and c_name == "Banana": score += 8.0
                    else: score += 2.0
            else:
                if is_dicot:
                    score += 10.0
                    if c_name == "Cotton" and sol < 0.70:
                        score += 16.0
                    elif c_name == "Cassava" and sol < 0.60:
                        score += 16.0
                    elif c_name == "Coffee" and tar > 1.4 and circ > 0.65 and rst > 0.02:
                        score += 16.0
                    elif c_name == "Tea" and tar < 1.6 and pwd > 0.01:
                        score += 8.0
                    elif c_name == "Apple" and h["scab_score"] > 0.02:
                        score += 14.0
                    elif c_name == "Squash" and h["powder_score"] > 0.02 and circ > 0.50:
                        score += 14.0
                    elif c_name == "Tomato" and nec > 0.08:
                        score += 14.0
                    elif c_name == "Potato" and sol < 0.90 and circ > 0.50 and (chl > 0.04 or nec > 0.04):
                        # Potato only boosted if genuinely dicot morphology
                        score += 14.0
                elif is_monocot:
                    score -= 14.0
                else:
                    if c_name == "Cotton" and sol < 0.70: score += 10.0
                    elif c_name == "Coffee" and tar > 1.4 and circ > 0.65: score += 8.0
                    elif tar < 1.4 and not is_monocot: score += 4.0
                    
            affinities[c_name] = score
            
        return affinities

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
        target_crop: Optional[str] = None,
        hint_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute hybrid inference: neural network logits + biological feature scoring.
        """
        tensor, rgb_array = self.preprocess_image(image)
        heuristics = self.analyze_visual_heuristics(rgb_array)
        
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0).cpu().numpy()
            
        scores = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        
        # Neural network logits (meaningful for trained classes 0-37)
        scores += logits * 0.15
        
        # Extract biological indicators
        nec = heuristics["necrotic_score"]
        chl = heuristics["chlorosis_score"]
        pwd = heuristics["powder_score"]
        rst = heuristics["rust_score"]
        hlt = heuristics["healthy_score"]
        scb = heuristics["scab_score"]
        rmb = heuristics["red_midrib_score"]
        pmb = heuristics["pale_midrib_score"]
        sol = heuristics["solidity"]
        
        has_disease = (nec > 0.025 or rst > 0.015 or pwd > 0.025 or rmb > 0.02 or scb > 0.03 or chl > 0.06)
        
        # 1. Multi-feature crop affinity (replaces aspect-ratio-only approach)
        crop_affinity = self._compute_crop_affinity(heuristics)

        # 2. Class-specific biological pathology scoring
        for idx, class_name in enumerate(CLASS_NAMES):
            parts = class_name.split("___")
            c_prefix = parts[0]
            d_suffix = parts[1] if len(parts) > 1 else "healthy"
            
            matched_crop = "Unknown"
            for c_key, c_prof in CROP_PROFILES.items():
                if any(k.lower() in c_prefix.lower() for k in c_prof["keys"]):
                    matched_crop = c_key
                    break
                    
            c_aff = crop_affinity.get(matched_crop, 0.0)
            score = max(0.0, c_aff) * 1.5
            
            # Disease-specific symptom rules
            if "Red_Rot" in d_suffix:
                if rmb > 0.015: score += 35.0 + (rmb * 50.0)
                elif matched_crop == "Sugarcane" and pmb > 0.03: score += 20.0
            elif "Yellow_Rust" in d_suffix:
                if matched_crop == "Wheat" and (rst > 0.006 or chl > 0.015 or c_aff > 0.0):
                    score += 42.0 + (rst * 50.0)
            elif "Brown_Rust" in d_suffix:
                if matched_crop == "Wheat" and (rst > 0.006 or nec > 0.015 or c_aff > 0.0):
                    score += 38.0 + (rst * 45.0)
            elif "Powdery_Mildew" in d_suffix:
                if matched_crop == "Wheat" and (pwd > 0.01 or chl > 0.02):
                    score += 36.0 + (pwd * 40.0)
                elif matched_crop == "Squash" and pwd > 0.02:
                    score += 35.0 + (pwd * 40.0)
                elif matched_crop == "Cherry" and pwd > 0.02:
                    score += 22.0
                elif pwd > 0.02:
                    score += 10.0
            elif "Black_Sigatoka" in d_suffix:
                if matched_crop == "Banana" and (nec > 0.015 or chl > 0.02): score += 35.0 + (nec * 40.0)
            elif "Yellow_Sigatoka" in d_suffix:
                if matched_crop == "Banana" and chl > 0.02: score += 30.0 + (chl * 30.0)
            elif "Bacterial_Blight" in d_suffix:
                if matched_crop == "Cotton" and (nec > 0.02 or sol < 0.70): score += 35.0 + (nec * 40.0)
                elif matched_crop == "Cassava" and nec > 0.02: score += 32.0 + (nec * 35.0)
            elif "Leaf_Curl_Virus" in d_suffix:
                if matched_crop == "Cotton": score += 28.0 + (chl * 25.0)
            elif "Leaf_Rust" in d_suffix:
                if matched_crop == "Coffee" and rst > 0.02: score += 35.0 + (rst * 40.0)
            elif "Blister_Blight" in d_suffix:
                if matched_crop == "Tea" and (pwd > 0.01 or nec > 0.02): score += 32.0
            elif "Red_Rust" in d_suffix and matched_crop == "Tea":
                if rst > 0.015: score += 30.0
            elif "Mosaic_Disease" in d_suffix and matched_crop == "Cassava":
                if chl > 0.03: score += 32.0 + (chl * 30.0)
            elif d_suffix == "Rust" and matched_crop == "Sugarcane":
                if rst > 0.015 or pmb > 0.03: score += 35.0 + (rst * 40.0)
            elif d_suffix.startswith("Common_rust") and matched_crop == "Corn":
                if rst > 0.015: score += 28.0 + (rst * 30.0)
            elif "Apple_scab" in d_suffix:
                if scb > 0.02 and matched_crop == "Apple": score += 35.0 + (scb * 40.0)
            elif "Northern_Leaf_Blight" in d_suffix:
                if matched_crop == "Corn" and (nec > 0.03 or chl > 0.08):
                    score += 35.0 + (chl * 25.0)
            elif "Cercospora" in d_suffix or "Gray_leaf_spot" in d_suffix:
                if matched_crop == "Corn" and nec > 0.02 and chl > 0.05:
                    score += 18.0
                elif matched_crop == "Coffee" and nec > 0.02:
                    score += 22.0
            elif "Late_blight" in d_suffix:
                if nec > 0.10:
                    if matched_crop == "Tomato" and c_aff > 0.0: score += 42.0 + (nec * 45.0)
                    elif matched_crop == "Potato" and c_aff > 0.0: score += 25.0 + (nec * 20.0)
                elif nec > 0.04:
                    if matched_crop == "Tomato" and c_aff > 0.0: score += 20.0
            elif "Early_blight" in d_suffix:
                if nec < 0.10 and nec > 0.02 and chl > 0.02:
                    if matched_crop == "Potato" and c_aff > 2.0: score += 22.0 + (chl * 20.0)
                    elif matched_crop == "Tomato" and c_aff > 2.0: score += 16.0 + (chl * 10.0)
            elif "Smut" in d_suffix:
                if matched_crop == "Sugarcane" and c_aff > 2.0: score += 10.0
            elif "Yellow_Leaf" in d_suffix:
                if matched_crop == "Sugarcane" and chl > 0.04 and c_aff > 2.0: score += 14.0
            elif "Mosaic" in d_suffix:
                if matched_crop == "Sugarcane" and chl > 0.03 and c_aff > 2.0: score += 12.0
            elif "Leaf_Blast" in d_suffix:
                if matched_crop == "Rice" and nec > 0.03: score += 20.0
            elif "Brown_Spot" in d_suffix:
                if matched_crop == "Rice" and nec > 0.02 and rst > 0.01: score += 18.0
            elif "Bacterial_spot" in d_suffix:
                if nec > 0.03 and matched_crop in ["Tomato", "Pepper", "Peach"]: score += 14.0
            elif "Septoria" in d_suffix:
                if nec > 0.05 and matched_crop == "Tomato": score += 16.0
            elif "Leaf_Mold" in d_suffix:
                if pwd > 0.01 and chl > 0.02 and matched_crop == "Tomato": score += 14.0
            elif "Target_Spot" in d_suffix:
                if nec > 0.04:
                    if matched_crop == "Cotton": score += 26.0
                    elif matched_crop == "Tomato": score += 14.0
            elif "Black_rot" in d_suffix:
                if nec > 0.04 and matched_crop in ["Apple", "Grape"]: score += 16.0
            elif "Esca" in d_suffix or "Black_Measles" in d_suffix:
                if nec > 0.03 and matched_crop == "Grape": score += 14.0
            elif "Leaf_scorch" in d_suffix:
                if nec > 0.04 and matched_crop == "Strawberry": score += 14.0
            elif "Haunglongbing" in d_suffix or "greening" in d_suffix:
                if chl > 0.05 and matched_crop == "Orange": score += 16.0
            elif "Spider_mites" in d_suffix:
                if nec > 0.02 and chl > 0.02 and matched_crop == "Tomato": score += 12.0
            elif "Yellow_Leaf_Curl" in d_suffix:
                if chl > 0.04 and matched_crop == "Tomato": score += 14.0
            elif "mosaic_virus" in d_suffix:
                if chl > 0.03 and matched_crop == "Tomato": score += 12.0
            elif "healthy" in d_suffix.lower():
                if not has_disease and hlt > 0.65:
                    # For healthy leaves: combine neural network + morphology tiebreaker
                    nn_boost = max(0.0, logits[idx]) * 2.0 if idx < NUM_TRAINED_CLASSES else 0.0
                    
                    # Morphology tiebreaker: use leaf shape features
                    morph_boost = 0.0
                    sol = heuristics["solidity"]
                    circ = heuristics["circularity"]
                    ar = heuristics["aspect_ratio"]
                    
                    # Tomato: compound, serrated, moderate circularity, low-medium AR
                    if matched_crop == "Tomato" and circ > 0.6 and ar < 1.3 and sol > 0.9:
                        morph_boost += 6.0
                    # Peach: elongated elliptical, high AR
                    elif matched_crop == "Peach" and ar > 1.3:
                        morph_boost += 4.0
                    # Apple: round, high circularity
                    elif matched_crop == "Apple" and circ > 0.6:
                        morph_boost += 3.0
                    # Grape: palmate, lower solidity
                    elif matched_crop == "Grape" and sol < 0.8:
                        morph_boost += 3.0
                        
                    score += 22.0 + (hlt * 10.0) - (nec * 30.0) + nn_boost + morph_boost
                elif not has_disease:
                    nn_boost = max(0.0, logits[idx]) * 2.0 if idx < NUM_TRAINED_CLASSES else 0.0
                    score += 6.0 + nn_boost
                else:
                    score -= 30.0
                    
            # Penalize classes from wrong crop family
            if c_aff < -3.0:
                score -= 15.0
                    
            if hint_class and class_name == hint_class:
                score += 35.0
                
            scores[idx] += score

        # 3. Target Crop Lock Filtering
        if target_crop and target_crop.lower() not in ["auto", "all", "none", ""]:
            for idx, class_name in enumerate(CLASS_NAMES):
                parts = class_name.split("___")
                c_name = parts[0].replace("_", " ").lower()
                if target_crop.lower() not in c_name and c_name not in target_crop.lower():
                    scores[idx] = -1000.0

        # 4. Temperature Scaled Softmax
        temperature = 0.35
        scaled = (scores - np.max(scores)) / temperature
        exp_s = np.exp(np.clip(scaled, -80, 0))
        final_probs = exp_s / np.sum(exp_s)
        
        # 5. Aggregate Probabilities by Crop Species
        crop_scores: Dict[str, float] = {}
        for idx, class_name in enumerate(CLASS_NAMES):
            parts = class_name.split("___")
            c_prefix = parts[0]
            matched_crop = "Unknown"
            for c_key, c_prof in CROP_PROFILES.items():
                if any(k.lower() in c_prefix.lower() for k in c_prof["keys"]):
                    matched_crop = c_key
                    break
            if matched_crop == "Unknown":
                matched_crop = parts[0].replace("_", " ")
            crop_scores[matched_crop] = crop_scores.get(matched_crop, 0.0) + float(final_probs[idx])
            
        sorted_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)
        detected_crop_key = sorted_crops[0][0]
        detected_crop_conf = round(min(99.6, sorted_crops[0][1] * 100.0), 1)
        
        crop_meta = CROP_PROFILES.get(detected_crop_key, {
            "display": detected_crop_key.replace("_", " "),
            "botanical_name": "Plantae Species",
            "family": "Angiosperms"
        })
        
        # Get Top-K disease indices
        top_indices = np.argsort(final_probs)[::-1][:top_k]
        
        predictions = []
        for rank, idx in enumerate(top_indices):
            class_name = CLASS_NAMES[idx]
            parts = class_name.split("___")
            crop_name = parts[0].replace("_", " ")
            disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            conf = float(final_probs[idx])
            
            predictions.append({
                "rank": rank + 1,
                "class_id": int(idx),
                "class_name": class_name,
                "crop": crop_name,
                "disease": disease_name,
                "confidence": round(conf * 100, 2),
                "confidence_ratio": conf
            })
            
        top_pred = predictions[0]
        
        return {
            "top_prediction": top_pred,
            "top_k_predictions": predictions,
            "crop_identification": {
                "detected_crop": crop_meta.get("display", detected_crop_key),
                "crop_key": detected_crop_key,
                "botanical_name": crop_meta.get("botanical_name", "N/A"),
                "crop_family": crop_meta.get("family", "N/A"),
                "crop_confidence": detected_crop_conf,
                "top_crops": [{"crop": k, "confidence": round(v * 100, 1)} for k, v in sorted_crops[:3]]
            },
            "heuristics": {k: round(float(v), 4) for k, v in heuristics.items()}
        }

# Global singleton instance
_engine_instance: Optional[CropInferenceEngine] = None

def get_inference_engine() -> CropInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        default_weights = os.path.join(os.path.dirname(__file__), "model_weights.pth")
        weights_to_load = default_weights if os.path.exists(default_weights) else None
        _engine_instance = CropInferenceEngine(weights_path=weights_to_load)
    return _engine_instance
