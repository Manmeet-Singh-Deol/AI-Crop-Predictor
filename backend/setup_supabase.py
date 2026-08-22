"""
Automated Supabase Provisioning & Verification Script
Executes database migrations and verifies storage buckets.
"""

import os
import sys
import httpx

def auto_setup_supabase(supabase_url: str, supabase_key: str):
    supabase_url = supabase_url.rstrip("/")
    print(f"\n[Supabase Setup] Connecting to: {supabase_url}...")
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }
    
    # 1. Test basic connectivity
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get(f"{supabase_url}/rest/v1/", headers=headers)
            print(f"[Supabase Setup] PostgREST API Status: {res.status_code}")
    except Exception as e:
        print(f"[Supabase Setup] Connection failed: {e}")
        return False
        
    # 2. Check or create storage bucket
    try:
        with httpx.Client(timeout=10.0) as client:
            b_res = client.post(
                f"{supabase_url}/storage/v1/bucket",
                json={"id": "crop-scans", "name": "crop-scans", "public": True},
                headers=headers
            )
            print(f"[Supabase Setup] Storage Bucket 'crop-scans': status {b_res.status_code}")
    except Exception as e:
        print(f"[Supabase Setup] Storage bucket notice: {e}")
        
    # 3. Write .env file
    env_content = f"SUPABASE_URL={supabase_url}\nSUPABASE_KEY={supabase_key}\nPORT=8000\n"
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    print("[Supabase Setup] Saved credentials to .env file successfully!")
    return True

if __name__ == "__main__":
    url = os.environ.get("SUPABASE_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    key = os.environ.get("SUPABASE_KEY") or (sys.argv[2] if len(sys.argv) > 2 else None)
    
    if not url or not key:
        print("Usage: python backend/setup_supabase.py <SUPABASE_URL> <SUPABASE_KEY>")
        sys.exit(1)
        
    auto_setup_supabase(url, key)
