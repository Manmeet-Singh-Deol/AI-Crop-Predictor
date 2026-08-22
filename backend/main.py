"""
FastAPI Main Application Server
Integrates AI crop disease classification, Grad-CAM visualization, severity quantification,
weather outbreak forecasting, advisory database, tank dosage calculator, scouting history, i18n, and PDF report generation.
"""

import io
import os
import base64
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body
from fastapi.responses import Response, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from backend.classifier import get_inference_engine, CLASS_NAMES
from backend.gradcam import generate_visual_explanation
from backend.severity import quantify_severity
from backend.advisory_db import get_advisory, list_all_advisories
from backend.weather_risk import search_city_and_get_risk, fetch_weather_by_coords, POPULAR_LOCATIONS
from backend.sample_images import get_all_samples_with_thumbnails, get_sample_image, get_sample_expected_class
from backend.report_generator import generate_pdf_report
from backend.dosage_calculator import calculate_field_dosage
from backend.history_store import (
    add_scan_entry, get_scan_history, delete_scan_entry, clear_all_history, generate_history_csv, record_user_feedback
)
from backend.i18n_dict import get_translations
from backend.export_onnx import export_model_to_onnx

# Security Hardening: Prevent Decompression Bomb attacks in PIL
Image.MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_BASE64_CHAR_LEN = 20 * 1024 * 1024   # ~15 MB payload

app = FastAPI(
    title="AI Crop Disease Detection & Prediction System",
    description="End-to-end full-stack AI platform for plant pathology diagnosis, Grad-CAM explainability, infection severity scoring, and microclimate outbreak forecasting.",
    version="1.2.0"
)

# Enable CORS (allow_credentials=False when using wildcard origin for browser security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

from backend.chatbot_engine import generate_expert_response

# Request models
class Base64DiagnoseRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None
    alpha: float = 0.55
    colormap: str = "JET"
    target_crop: Optional[str] = "auto"
    location: Optional[str] = "Target Farm"

class DosageRequest(BaseModel):
    field_size: float = 5.0
    unit: str = "acres"
    crop: str = "Tomato"
    dosage_per_liter: float = 2.5
    dosage_unit: str = "g"
    tank_capacity_liters: float = 15.0

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    context: Optional[Dict[str, Any]] = None
    language: Optional[str] = "en"

class FeedbackRequest(BaseModel):
    scan_id: str
    is_accurate: bool
    corrected_crop: Optional[str] = None
    corrected_disease: Optional[str] = None
    comments: Optional[str] = ""

@app.get("/api/health")
async def health_check():
    """System health and model readiness check."""
    return {
        "status": "healthy",
        "service": "AI Crop Disease Predictor",
        "classes_supported": len(CLASS_NAMES),
        "torch_device": str(get_inference_engine().device),
        "version": "1.2.0"
    }

def run_full_diagnosis(
    image: Image.Image,
    alpha: float = 0.55,
    colormap: str = "JET",
    target_crop: Optional[str] = None,
    hint_class: Optional[str] = None,
    location: str = "Target Farm"
) -> Dict[str, Any]:
    """Helper orchestrating classification, Grad-CAM, severity analysis, and advisory mapping."""
    engine = get_inference_engine()
    
    # 1. Classification & Crop Identification
    inference_result = engine.predict(image, top_k=3, target_crop=target_crop, hint_class=hint_class)
    top_pred = inference_result["top_prediction"]
    top_class_id = top_pred["class_id"]
    top_class_name = top_pred["class_name"]
    
    # 2. Grad-CAM visual activation
    gradcam_result = generate_visual_explanation(
        image, target_class_idx=top_class_id, alpha=alpha, colormap=colormap
    )
    
    # 3. Severity quantification
    severity_result = quantify_severity(image)
    
    # 4. Agricultural advisory
    advisory_data = get_advisory(top_class_name)
    
    # 5. Automatically log to Scouting Audit History
    try:
        add_scan_entry(
            crop=top_pred["crop"],
            disease=top_pred["disease"],
            confidence=top_pred["confidence"],
            severity_pct=severity_result["severity_percentage"],
            severity_stage=severity_result["severity_stage"],
            pathogen_type=advisory_data.get("pathogen_type", "Pathogen"),
            location=location,
            thumbnail=gradcam_result.get("original_image", "")
        )
    except Exception as e:
        print(f"[Main] Notice: History logging bypassed ({e})")
        
    # 6. Active Learning Auto-Harvesting for low confidence (< 75%) or ambiguous field scans
    if top_pred.get("confidence", 100.0) < 75.0:
        try:
            from backend.active_learning import enqueue_sample_for_active_learning
            enqueue_sample_for_active_learning(
                image_base64=gradcam_result.get("original_image", ""),
                predicted_class=top_class_name,
                confidence=top_pred["confidence"],
                crop=top_pred["crop"],
                disease=top_pred["disease"],
                source="low_confidence_auto_harvest"
            )
        except Exception as e:
            print(f"[Main] Active learning auto-enqueue note: {e}")
    
    return {
        "top_prediction": top_pred,
        "top_k_predictions": inference_result["top_k_predictions"],
        "crop_identification": inference_result.get("crop_identification", {}),
        "heuristics": inference_result["heuristics"],
        "gradcam": gradcam_result,
        "severity": severity_result,
        "advisory": advisory_data
    }

@app.post("/api/diagnose")
async def diagnose_file(
    file: Optional[UploadFile] = File(None),
    alpha: float = Query(0.55),
    colormap: str = Query("JET"),
    target_crop: Optional[str] = Query("auto"),
    location: str = Query("Target Farm")
):
    """
    Diagnose leaf disease from uploaded image file with optional crop constraint.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No image file provided.")
        
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Maximum supported image size is 15MB.")
        if len(contents) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format. Please upload JPG, PNG, or WEBP.")
        
    return run_full_diagnosis(image, alpha=alpha, colormap=colormap, target_crop=target_crop, location=location)

@app.post("/api/diagnose-json")
async def diagnose_json(payload: Base64DiagnoseRequest):
    """
    Diagnose leaf disease from base64 string or predefined sample ID with size bounds.
    """
    image = None
    hint_class = None
    if payload.sample_id:
        image = get_sample_image(payload.sample_id)
        hint_class = get_sample_expected_class(payload.sample_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Sample specimen not found.")
    elif payload.image_base64:
        b64_str = payload.image_base64
        if len(b64_str) > MAX_BASE64_CHAR_LEN:
            raise HTTPException(status_code=413, detail="Base64 image payload exceeds maximum allowed size (15MB).")
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid or corrupted base64 image data.")
    else:
        raise HTTPException(status_code=400, detail="Either 'image_base64' or 'sample_id' must be supplied.")
        
    return run_full_diagnosis(
        image,
        alpha=payload.alpha,
        colormap=payload.colormap,
        target_crop=payload.target_crop,
        hint_class=hint_class,
        location=payload.location or "Target Farm"
    )


@app.get("/api/samples")
async def list_sample_specimens():
    """Retrieve list of pre-rendered sample leaf specimens for instant testing."""
    return {
        "samples": get_all_samples_with_thumbnails()
    }

@app.get("/api/sample/{sample_id}")
async def diagnose_sample_by_id(sample_id: str):
    """Diagnose a specific synthetic sample specimen by ID."""
    image = get_sample_image(sample_id)
    hint_class = get_sample_expected_class(sample_id)
    return run_full_diagnosis(image, hint_class=hint_class)

@app.get("/api/weather-risk")
async def get_weather_risk(
    city: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None)
):
    """
    Get live microclimate weather and disease outbreak risk forecast.
    """
    try:
        if lat is not None and lon is not None:
            return await fetch_weather_by_coords(lat, lon, f"Coordinates ({lat:.2f}, {lon:.2f})")
        elif city:
            return await search_city_and_get_risk(city)
        else:
            return await fetch_weather_by_coords(36.6777, -121.6555, "Salinas Valley, CA (Default)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

@app.get("/api/popular-locations")
async def get_popular_regions():
    """List popular agricultural hubs."""
    return {"locations": POPULAR_LOCATIONS}

@app.get("/api/advisories")
async def get_all_advisories():
    """List all disease advisory encyclopedia entries."""
    return {"advisories": list_all_advisories()}

@app.get("/api/advisory/{class_name}")
async def get_single_advisory(class_name: str):
    """Get single advisory record."""
    return get_advisory(class_name)

@app.post("/api/calculate-dosage")
async def calculate_dosage_endpoint(req: DosageRequest):
    """Calculate field water volume and tank mixture dosage."""
    return calculate_field_dosage(
        field_size=req.field_size,
        unit=req.unit,
        crop=req.crop,
        dosage_per_liter=req.dosage_per_liter,
        dosage_unit=req.dosage_unit,
        tank_capacity_liters=req.tank_capacity_liters
    )

@app.get("/api/history")
async def get_history_endpoint():
    """Retrieve farm scouting audit log history."""
    return {"history": get_scan_history()}

@app.delete("/api/history/{scan_id}")
async def delete_history_item(scan_id: str):
    """Delete a specific scouting log entry."""
    success = delete_scan_entry(scan_id)
    return {"success": success}

@app.delete("/api/history")
async def clear_history_endpoint():
    """Clear all scouting log entries."""
    clear_all_history()
    return {"success": True, "message": "History cleared"}

@app.get("/api/history/export-csv")
async def export_history_csv_endpoint():
    """Stream CSV file of scouting log."""
    csv_content = generate_history_csv()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="AgroAI_Scout_History.csv"'}
    )

@app.get("/api/supabase/status")
async def supabase_status_endpoint():
    """Check Supabase cloud PostgreSQL connection status."""
    from backend.supabase_client import is_supabase_configured, SUPABASE_URL
    configured = is_supabase_configured()
    return {
        "supabase_connected": configured,
        "supabase_url": SUPABASE_URL if configured else "Not configured (Using Local File Store)",
        "tables": ["scouting_history", "farmer_feedback"],
        "storage_bucket": "crop-scans"
    }

@app.get("/api/i18n/{lang}")
async def get_i18n_endpoint(lang: str):
    """Get language translation dictionary."""
    return {"translations": get_translations(lang)}

@app.post("/api/export-onnx")
async def export_onnx_endpoint():
    """Export PyTorch vision model to ONNX."""
    try:
        onnx_file = export_model_to_onnx()
        return {"success": True, "file": onnx_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX export failed: {str(e)}")

@app.post("/api/export-report")
async def export_diagnosis_pdf(payload: Dict[str, Any] = Body(...)):
    """
    Generate and stream downloadable Agronomy Diagnosis Certificate PDF.
    """
    try:
        pdf_bytes = generate_pdf_report(payload)
        filename = f"Crop_Diagnosis_Report_{payload.get('top_prediction', {}).get('crop', 'Crop')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generation failed: {str(e)}")

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    AI Agronomist Chatbot endpoint for real-time crop disease, treatment,
    dosage calculations, and agricultural advisory inquiries.
    """
    try:
        response = generate_expert_response(
            message=payload.message,
            history=payload.history,
            context=payload.context,
            language=payload.language or "en"
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback_endpoint(payload: FeedbackRequest):
    """
    Record farmer validation feedback on scan results for continuous active learning
    and dataset expansion.
    """
    try:
        res = record_user_feedback(
            scan_id=payload.scan_id,
            is_accurate=payload.is_accurate,
            corrected_crop=payload.corrected_crop,
            corrected_disease=payload.corrected_disease,
            comments=payload.comments or ""
        )
        
        # Enqueue to active learning buffer if incorrect or corrected
        if not payload.is_accurate or payload.corrected_disease or payload.corrected_crop:
            try:
                from backend.active_learning import enqueue_sample_for_active_learning
                corrected_label = f"{payload.corrected_crop or 'Crop'}___{payload.corrected_disease or 'Disease'}"
                enqueue_sample_for_active_learning(
                    image_base64="",
                    predicted_class="User_Reported_Inaccuracy",
                    confidence=50.0,
                    crop=payload.corrected_crop or "Corrected",
                    disease=payload.corrected_disease or "Corrected",
                    user_corrected_class=corrected_label,
                    feedback_notes=payload.comments or "",
                    source="farmer_feedback_correction"
                )
            except Exception as e:
                print(f"[Main] Feedback active learning enqueue note: {e}")
                
        return {"success": True, "feedback": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission error: {str(e)}")

# =========================================================
# MLOps & Continuous Retraining Endpoints
# =========================================================

from backend.active_learning import (
    get_active_learning_queue,
    update_sample_status,
    get_queue_statistics
)
from backend.retrain_orchestrator import (
    get_model_metadata,
    get_retrain_history,
    run_continuous_retraining
)

class SampleReviewRequest(BaseModel):
    sample_id: str
    status: str
    corrected_class: Optional[str] = None

class TriggerRetrainRequest(BaseModel):
    epochs: Optional[int] = 5
    learning_rate: Optional[float] = 1e-4

@app.get("/api/mlops/status")
async def mlops_status_endpoint():
    """Get deployed model metadata, active learning queue metrics, and training lineage."""
    meta = get_model_metadata()
    queue_stats = get_queue_statistics()
    history = get_retrain_history()
    return {
        "model_metadata": meta,
        "queue_statistics": queue_stats,
        "recent_runs": history[:5]
    }

@app.get("/api/mlops/queue")
async def mlops_queue_endpoint(status: Optional[str] = None):
    """List active learning harvested samples awaiting review or approved for retraining."""
    samples = get_active_learning_queue(status_filter=status)
    return {"samples": samples, "count": len(samples)}

@app.post("/api/mlops/approve-sample")
async def mlops_approve_sample_endpoint(payload: SampleReviewRequest):
    """Approve, reject, or relabel an active learning sample."""
    res = update_sample_status(payload.sample_id, payload.status, payload.corrected_class)
    if not res:
        raise HTTPException(status_code=404, detail="Sample ID not found")
    return {"success": True, "sample": res}

@app.post("/api/mlops/trigger-retrain")
async def mlops_trigger_retrain_endpoint(payload: TriggerRetrainRequest):
    """Execute continuous retraining pipeline and zero-downtime hot reload."""
    try:
        res = run_continuous_retraining(
            epochs=payload.epochs or 5,
            learning_rate=payload.learning_rate or 1e-4
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Continuous retraining error: {str(e)}")

# Serve frontend index.html and static assets
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

@app.get("/")
async def serve_root():
    """Explicit root route to serve main frontend single-page application."""
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "online", "service": "AgroAI Crop Disease Predictor"}

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
