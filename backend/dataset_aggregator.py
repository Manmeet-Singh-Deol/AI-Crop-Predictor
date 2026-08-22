"""
Multi-Dataset Agricultural Vision Aggregator for AgroAI Platform
Consolidates diverse open-access agricultural pathology datasets:
- PlantVillage (54k images - Lab baseline)
- PlantDoc (In-field real farm photos)
- Paddy Doctor (Rice in-field diseases)
- Cassava Leaf Disease (Smartphone field captures)
- Sugarcane Disease Dataset (Cash crop field samples)
- RoCoLe Coffee Leaf Dataset (Tropical plantation photos)
- Cotton & Banana Disease Repositories
Maps heterogeneous folder nomenclatures into the unified 67-class AgroAI taxonomy.
"""

import os
import shutil
import urllib.request
import zipfile
import tarfile
from typing import Dict, List, Tuple, Optional

from backend.classifier import CLASS_NAMES

# Direct open-access dataset mirrors
DATASET_MIRRORS = {
    "plantvillage": {
        "name": "PlantVillage Benchmark",
        "url": "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
        "subpath": "PlantVillage-Dataset-master/raw/color"
    },
    "plantdoc": {
        "name": "PlantDoc In-Field Real World Dataset",
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip",
        "subpath": "PlantDoc-Dataset-master/train"
    }
}

# Mapping table from external dataset class names to unified AgroAI 67-class taxonomy
CLASS_ALIAS_MAP = {
    # PlantDoc mappings
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf early blight": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Potato leaf": "Potato___healthy",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Apple leaf": "Apple___healthy",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Grape leaf": "Grape___healthy",
    "Grape leaf black rot": "Grape___Black_rot",
    "Strawberry leaf": "Strawberry___healthy",
    
    # Paddy Doctor mappings
    "bacterial_leaf_blight": "Rice___Brown_Spot",
    "blast": "Rice___Leaf_Blast",
    "brown_spot": "Rice___Brown_Spot",
    "normal": "Rice___healthy",
    
    # Sugarcane mappings
    "red_rot": "Sugarcane___Red_Rot",
    "rust": "Sugarcane___Rust",
    "yellow_leaf": "Sugarcane___Yellow_Leaf",
    "mosaic": "Sugarcane___Mosaic",
    "smut": "Sugarcane___Smut",
    "bacterial_blight": "Sugarcane___Bacterial_Blight",
    
    # Cassava mappings
    "cbb": "Cassava___Bacterial_Blight",
    "cmd": "Cassava___Mosaic_Disease",
    "healthy_cassava": "Cassava___healthy",
    
    # Coffee mappings
    "rust_coffee": "Coffee___Leaf_Rust",
    "cercospora_coffee": "Coffee___Cercospora_Leaf_Spot",
    "healthy_coffee": "Coffee___healthy",
    
    # Wheat mappings
    "yellow_rust": "Wheat___Yellow_Rust",
    "brown_rust": "Wheat___Brown_Rust",
    "powdery_mildew": "Wheat___Powdery_Mildew",
    "healthy_wheat": "Wheat___healthy",
    
    # Cotton mappings
    "bacterial_blight_cotton": "Cotton___Bacterial_Blight",
    "curl_virus": "Cotton___Leaf_Curl_Virus",
    "target_spot": "Cotton___Target_Spot",
    "healthy_cotton": "Cotton___healthy",
    
    # Banana mappings
    "black_sigatoka": "Banana___Black_Sigatoka",
    "yellow_sigatoka": "Banana___Yellow_Sigatoka",
    "healthy_banana": "Banana___healthy"
}

def resolve_target_class(raw_folder_name: str) -> Optional[str]:
    """Resolve raw dataset class name to canonical 67-class AgroAI taxonomy."""
    # 1. Exact match
    if raw_folder_name in CLASS_NAMES:
        return raw_folder_name
        
    # 2. Known alias map
    if raw_folder_name in CLASS_ALIAS_MAP:
        return CLASS_ALIAS_MAP[raw_folder_name]
        
    # 3. Fuzzy normalizer
    norm = raw_folder_name.lower().replace("-", "_").replace(" ", "_")
    for alias, target in CLASS_ALIAS_MAP.items():
        if alias.lower().replace("-", "_").replace(" ", "_") in norm:
            return target
            
    for cls in CLASS_NAMES:
        cls_norm = cls.lower().replace("-", "_").replace(" ", "_")
        if norm in cls_norm or cls_norm in norm:
            return cls
            
    return None

def merge_datasets_into_unified_structure(
    source_dirs: List[str],
    output_dataset_dir: str,
    train_ratio: float = 0.8
) -> Dict[str, int]:
    """
    Merge multiple raw dataset folders into unified train/val structure.
    """
    os.makedirs(os.path.join(output_dataset_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, "val"), exist_ok=True)
    
    # Initialize class directories
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(output_dataset_dir, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(output_dataset_dir, "val", cls), exist_ok=True)
        
    stats = {cls: 0 for cls in CLASS_NAMES}
    
    for sdir in source_dirs:
        if not os.path.exists(sdir):
            continue
            
        for root, dirs, files in os.walk(sdir):
            if not files:
                continue
            folder_name = os.path.basename(root)
            target_class = resolve_target_class(folder_name)
            
            if not target_class or target_class not in CLASS_NAMES:
                continue
                
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            split_idx = int(len(img_files) * train_ratio)
            
            for idx, img_name in enumerate(img_files):
                src_path = os.path.join(root, img_name)
                split = "train" if idx < split_idx else "val"
                dst_name = f"{os.path.basename(sdir)}_{idx}_{img_name}"
                dst_path = os.path.join(output_dataset_dir, split, target_class, dst_name)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    stats[target_class] += 1
                except Exception:
                    pass
                    
    print(f"[Dataset Aggregator] Successfully aggregated {sum(stats.values())} images across {len([k for k, v in stats.items() if v > 0])} classes.")
    return stats
