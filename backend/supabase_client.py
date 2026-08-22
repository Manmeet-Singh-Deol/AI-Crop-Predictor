"""
Supabase Cloud Database & Storage Connector for AgroAI Platform
Provides persistent PostgreSQL cloud storage for scouting scans, farmer feedback,
and leaf image asset storage in Supabase Storage.
"""

import os
import io
import base64
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List

def _load_env_file():
    """Load key-value pairs from .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

def get_supabase_url() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/")

def get_supabase_key() -> str:
    return os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")

def is_supabase_configured() -> bool:
    """Check whether Supabase environment variables are present and non-empty."""
    return bool(get_supabase_url() and get_supabase_key())

def _get_headers() -> Dict[str, str]:
    """Generate standard Supabase authentication and PostgREST headers."""
    key = get_supabase_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def upload_leaf_to_supabase_storage(image_b64: str, file_name: str) -> Optional[str]:
    """
    Upload a base64 leaf image to Supabase Storage 'crop-scans' bucket.
    Returns the public CDN image URL or None on failure.
    """
    if not is_supabase_configured() or not image_b64:
        return None
        
    url = get_supabase_url()
    key = get_supabase_key()
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        raw_bytes = base64.b64decode(image_b64)
        
        storage_url = f"{url}/storage/v1/object/crop-scans/{file_name}"
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "image/jpeg",
            "x-upsert": "true"
        }
        
        with httpx.Client(timeout=8.0) as client:
            res = client.post(storage_url, content=raw_bytes, headers=headers)
            if res.status_code in [200, 201]:
                return f"{url}/storage/v1/object/public/crop-scans/{file_name}"
            else:
                print(f"[Supabase Storage] Notice: upload status {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[Supabase Storage] Upload error: {e}")
        
    return None

def sync_scan_to_supabase(scan_data: Dict[str, Any], thumbnail_b64: Optional[str] = None) -> bool:
    """
    Insert a diagnosis scan record into the Supabase 'scouting_history' PostgreSQL table.
    """
    if not is_supabase_configured():
        return False
        
    url = get_supabase_url()
    try:
        image_url = None
        if thumbnail_b64 and len(thumbnail_b64) > 50:
            scan_id = scan_data.get("id", f"scan_{int(datetime.now().timestamp())}")
            image_url = upload_leaf_to_supabase_storage(thumbnail_b64, f"{scan_id}.jpg")
            
        row = {
            "scan_id": scan_data.get("id"),
            "created_at": datetime.now().isoformat(),
            "crop": scan_data.get("crop"),
            "disease": scan_data.get("disease"),
            "confidence": float(scan_data.get("confidence", 0.0)),
            "severity_percentage": float(scan_data.get("severity_percentage", 0.0)),
            "severity_stage": scan_data.get("severity_stage", "Low"),
            "pathogen_type": scan_data.get("pathogen_type", "Unknown"),
            "location": scan_data.get("location", "Target Farm"),
            "image_url": image_url or scan_data.get("thumbnail", "")
        }
        
        endpoint = f"{url}/rest/v1/scouting_history"
        with httpx.Client(timeout=6.0) as client:
            res = client.post(endpoint, json=row, headers=_get_headers())
            if res.status_code in [200, 201]:
                return True
            else:
                print(f"[Supabase DB] Sync error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[Supabase DB] Error syncing scan: {e}")
        
    return False

def fetch_scans_from_supabase(limit: int = 100) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch recent scouting scan records from the Supabase 'scouting_history' table.
    """
    if not is_supabase_configured():
        return None
        
    url = get_supabase_url()
    try:
        endpoint = f"{url}/rest/v1/scouting_history?select=*&order=created_at.desc&limit={limit}"
        with httpx.Client(timeout=6.0) as client:
            res = client.get(endpoint, headers=_get_headers())
            if res.status_code == 200:
                rows = res.json()
                results = []
                for r in rows:
                    results.append({
                        "id": r.get("scan_id"),
                        "timestamp": r.get("created_at", "")[:19].replace("T", " "),
                        "crop": r.get("crop"),
                        "disease": r.get("disease"),
                        "confidence": r.get("confidence"),
                        "severity_percentage": r.get("severity_percentage"),
                        "severity_stage": r.get("severity_stage"),
                        "pathogen_type": r.get("pathogen_type"),
                        "location": r.get("location"),
                        "thumbnail": r.get("image_url", "")
                    })
                return results
            else:
                print(f"[Supabase DB] Fetch error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[Supabase DB] Error fetching history: {e}")
        
    return None

def delete_scan_from_supabase(scan_id: str) -> bool:
    """Delete a scan record from Supabase by scan_id."""
    if not is_supabase_configured():
        return False
        
    url = get_supabase_url()
    try:
        endpoint = f"{url}/rest/v1/scouting_history?scan_id=eq.{scan_id}"
        with httpx.Client(timeout=6.0) as client:
            res = client.delete(endpoint, headers=_get_headers())
            return res.status_code in [200, 204]
    except Exception as e:
        print(f"[Supabase DB] Error deleting scan: {e}")
        return False

def sync_feedback_to_supabase(fb_data: Dict[str, Any]) -> bool:
    """Insert farmer validation feedback into the Supabase 'farmer_feedback' table."""
    if not is_supabase_configured():
        return False
        
    url = get_supabase_url()
    try:
        row = {
            "scan_id": fb_data.get("scan_id"),
            "created_at": datetime.now().isoformat(),
            "is_accurate": fb_data.get("is_accurate", True),
            "corrected_crop": fb_data.get("corrected_crop", ""),
            "corrected_disease": fb_data.get("corrected_disease", ""),
            "comments": fb_data.get("comments", "")
        }
        endpoint = f"{url}/rest/v1/farmer_feedback"
        with httpx.Client(timeout=6.0) as client:
            res = client.post(endpoint, json=row, headers=_get_headers())
            return res.status_code in [200, 201]
    except Exception as e:
        print(f"[Supabase DB] Error syncing feedback: {e}")
        return False
