"""
Field Scouting Diagnosis History & Audit Store
Persists farm inspection scans, GPS coordinates, severity metrics, and allows CSV exporting.
"""

import os
import json
import csv
import io
from datetime import datetime
from typing import List, Dict, Any, Optional

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scout_history.json")

def _load_raw_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[HistoryStore] Error loading history ({e}), resetting.")
        return []

def _save_raw_history(entries: List[Dict[str, Any]]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[HistoryStore] Error saving history: {e}")

def add_scan_entry(
    crop: str,
    disease: str,
    confidence: float,
    severity_pct: float,
    severity_stage: str,
    pathogen_type: str,
    location: str = "Target Farm",
    thumbnail: Optional[str] = None
) -> Dict[str, Any]:
    """Add a new diagnosis scan to the persistent history."""
    entries = _load_raw_history()
    
    new_entry = {
        "id": f"SCN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(entries)+1}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "crop": crop,
        "disease": disease,
        "confidence": round(float(confidence), 1),
        "severity_percentage": round(float(severity_pct), 1),
        "severity_stage": severity_stage,
        "pathogen_type": pathogen_type,
        "location": location,
        "thumbnail": thumbnail or ""
    }
    
    # Keep up to 100 most recent records
    entries.insert(0, new_entry)
    entries = entries[:100]
    _save_raw_history(entries)
    return new_entry

def get_scan_history() -> List[Dict[str, Any]]:
    """Retrieve all recorded farm scouting scans."""
    return _load_raw_history()

def delete_scan_entry(scan_id: str) -> bool:
    """Delete a specific scan record by ID."""
    entries = _load_raw_history()
    filtered = [e for e in entries if e.get("id") != scan_id]
    if len(filtered) < len(entries):
        _save_raw_history(filtered)
        return True
    return False

def clear_all_history() -> None:
    """Clear all stored scouting history."""
    _save_raw_history([])

def generate_history_csv() -> str:
    """Export all scan records as CSV string."""
    entries = _load_raw_history()
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow([
        "Scan ID", "Timestamp", "Location", "Crop", "Diagnosis",
        "Confidence (%)", "Severity (%)", "Severity Stage", "Pathogen Type"
    ])
    
    for e in entries:
        writer.writerow([
            e.get("id", ""),
            e.get("timestamp", ""),
            e.get("location", ""),
            e.get("crop", ""),
            e.get("disease", ""),
            e.get("confidence", 0.0),
            e.get("severity_percentage", 0.0),
            e.get("severity_stage", ""),
            e.get("pathogen_type", "")
        ])
        
    return output.getvalue()

def record_user_feedback(
    scan_id: str,
    is_accurate: bool,
    corrected_crop: Optional[str] = None,
    corrected_disease: Optional[str] = None,
    comments: str = ""
) -> Dict[str, Any]:
    """Record farmer validation feedback for continuous model retraining and active learning."""
    feedback_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "farmer_feedback.json")
    feedbacks = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        except Exception:
            feedbacks = []
            
    fb_entry = {
        "scan_id": scan_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_accurate": is_accurate,
        "corrected_crop": corrected_crop or "",
        "corrected_disease": corrected_disease or "",
        "comments": comments
    }
    feedbacks.append(fb_entry)
    try:
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[HistoryStore] Error saving feedback: {e}")
    return fb_entry

