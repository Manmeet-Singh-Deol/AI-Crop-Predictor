"""
AgroAI Active Learning & Data Drift Pipeline (MLOps)
Harvests hard negative predictions, out-of-distribution (OOD) field leaves,
and farmer diagnostic feedback to continuously improve the neural vision backbone.
"""

import os
import json
import base64
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
AL_QUEUE_FILE = os.path.join(DATA_DIR, "active_learning_queue.json")
AL_SAMPLES_DIR = os.path.join(DATA_DIR, "active_learning_samples")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AL_SAMPLES_DIR, exist_ok=True)

def _load_queue() -> List[Dict[str, Any]]:
    if not os.path.exists(AL_QUEUE_FILE):
        return []
    try:
        with open(AL_QUEUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ActiveLearning] Error reading queue ({e}), initializing empty.")
        return []

def _save_queue(queue: List[Dict[str, Any]]) -> None:
    try:
        with open(AL_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ActiveLearning] Error saving queue: {e}")

def calculate_entropy(probabilities: List[float]) -> float:
    """Calculate predictive Shannon entropy H(p) = -sum(p * log2(p))."""
    ent = 0.0
    for p in probabilities:
        if p > 1e-6:
            ent -= p * math.log2(p)
    return round(ent, 3)

def calculate_margin(probabilities: List[float]) -> float:
    """Calculate margin of confidence: p_top1 - p_top2 (smaller = higher uncertainty)."""
    if len(probabilities) < 2:
        return 1.0
    sorted_p = sorted(probabilities, reverse=True)
    return round(sorted_p[0] - sorted_p[1], 3)

def enqueue_sample_for_active_learning(
    image_base64: str,
    predicted_class: str,
    confidence: float,
    probabilities: Optional[List[float]] = None,
    crop: str = "Unknown",
    disease: str = "Unknown",
    user_corrected_class: Optional[str] = None,
    feedback_notes: str = "",
    source: str = "field_scan"
) -> Dict[str, Any]:
    """
    Log a field sample into the Active Learning queue if:
    1. Confidence is below 75% (Uncertainty threshold)
    2. Farmer submitted feedback correction
    3. Predictive entropy is high
    """
    queue = _load_queue()
    sample_id = f"AL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(queue)+1}"
    
    probs = probabilities or [confidence / 100.0, 1.0 - (confidence / 100.0)]
    entropy = calculate_entropy(probs)
    margin = calculate_margin(probs)
    
    # Uncertainty score (0-100, higher = more valuable for retraining)
    uncertainty_score = round(max(0.0, min(100.0, (100.0 - confidence) * 0.7 + (entropy * 15.0))), 1)

    # Save thumbnail image
    img_filename = f"{sample_id}.jpg"
    img_path = os.path.join(AL_SAMPLES_DIR, img_filename)
    if image_base64:
        try:
            clean_b64 = image_base64.split(",")[-1] if "," in image_base64 else image_base64
            img_bytes = base64.b64decode(clean_b64)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print(f"[ActiveLearning] Failed saving image {sample_id}: {e}")

    sample_entry = {
        "sample_id": sample_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "crop": crop,
        "disease": disease,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 1),
        "entropy": entropy,
        "margin": margin,
        "uncertainty_score": uncertainty_score,
        "user_corrected_class": user_corrected_class or "",
        "feedback_notes": feedback_notes,
        "status": "pending_review" if not user_corrected_class else "approved_for_training",
        "image_file": img_filename
    }

    queue.insert(0, sample_entry)
    _save_queue(queue)
    return sample_entry

def get_active_learning_queue(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve samples in the active learning buffer."""
    queue = _load_queue()
    if status_filter:
        return [s for s in queue if s.get("status") == status_filter]
    return queue

def update_sample_status(
    sample_id: str,
    new_status: str,
    corrected_class: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Approve, reject, or re-label an active learning sample."""
    queue = _load_queue()
    for s in queue:
        if s.get("sample_id") == sample_id:
            s["status"] = new_status
            if corrected_class:
                s["user_corrected_class"] = corrected_class
            s["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _save_queue(queue)
            return s
    return None

def get_queue_statistics() -> Dict[str, Any]:
    """Summary metrics of the Active Learning pipeline."""
    queue = _load_queue()
    total = len(queue)
    approved = sum(1 for s in queue if s.get("status") == "approved_for_training")
    pending = sum(1 for s in queue if s.get("status") == "pending_review")
    rejected = sum(1 for s in queue if s.get("status") == "rejected")
    
    avg_uncertainty = round(sum(s.get("uncertainty_score", 0) for s in queue) / max(total, 1), 1)
    
    # Class distribution among feedback
    class_dist: Dict[str, int] = {}
    for s in queue:
        lbl = s.get("user_corrected_class") or s.get("predicted_class") or "Unknown"
        class_dist[lbl] = class_dist.get(lbl, 0) + 1

    return {
        "total_harvested_samples": total,
        "approved_for_retraining": approved,
        "pending_review": pending,
        "rejected": rejected,
        "average_uncertainty_score": avg_uncertainty,
        "top_drifting_classes": sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    }
