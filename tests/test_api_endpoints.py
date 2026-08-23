"""
Integration API Test for FastAPI Endpoints
Tests live routing, JSON diagnostic requests, weather forecasting, PDF streaming,
dosage calculations, farm scouting audit log, i18n, and ONNX export.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app
from backend.dosage_calculator import calculate_field_dosage
from backend.history_store import add_scan_entry, get_scan_history, clear_all_history
from backend.i18n_dict import get_translations

class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["classes_supported"], 67)

    def test_02_samples_endpoint(self):
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("samples", data)
        self.assertGreater(len(data["samples"]), 0)
        sample_ids = [s["id"] for s in data["samples"]]
        self.assertIn("sample_sugarcane_red_rot", sample_ids)


    def test_03_diagnose_sample_json(self):
        payload = {
            "sample_id": "sample_tomato_late_blight",
            "alpha": 0.6,
            "colormap": "JET"
        }
        res = self.client.post("/api/diagnose-json", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("top_prediction", data)
        self.assertIn("gradcam", data)
        self.assertIn("severity", data)
        self.assertIn("advisory", data)
        self.assertEqual(data["top_prediction"]["crop"], "Tomato")

    def test_04_weather_risk_endpoint(self):
        res = self.client.get("/api/weather-risk?city=Salinas")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("epidemiological_risk", data)
        self.assertIn("five_day_forecast", data)

    def test_05_advisories_list(self):
        res = self.client.get("/api/advisories")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("advisories", data)
        self.assertGreaterEqual(len(data["advisories"]), 40)

    def test_06_pdf_export_endpoint(self):
        sample_res = self.client.get("/api/sample/sample_tomato_late_blight")
        diag_data = sample_res.json()
        diag_data["weather"] = {
            "epidemiological_risk": {
                "fungal_risk_score": 85.0,
                "threat_level": "High / Critical Risk",
                "overall_outbreak_risk": 78.5
            }
        }
        
        pdf_res = self.client.post("/api/export-report", json=diag_data)
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["content-type"], "application/pdf")
        self.assertGreater(len(pdf_res.content), 1000)

    def test_07_frontend_static_serving(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("AgroAI", res.text)

    def test_08_dosage_calculator_endpoint(self):
        payload = {
            "field_size": 10.0,
            "unit": "acres",
            "crop": "Tomato",
            "dosage_per_liter": 2.5,
            "dosage_unit": "g",
            "tank_capacity_liters": 15.0
        }
        res = self.client.post("/api/calculate-dosage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["total_water_liters"], 1000)
        self.assertIn("total_product_required", data)
        self.assertIn("num_tanks_required", data)

    def test_09_history_and_csv_export(self):
        clear_all_history()
        add_scan_entry("Tomato", "Late Blight", 95.5, 22.0, "Moderate", "Oomycete", "Field 4")
        
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data["history"]), 1)
        
        csv_res = self.client.get("/api/history/export-csv")
        self.assertEqual(csv_res.status_code, 200)
        self.assertEqual(csv_res.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("Tomato", csv_res.text)

    def test_10_i18n_endpoint(self):
        res = self.client.get("/api/i18n/hi")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("translations", data)
        self.assertIn("एग्रो-एआई", data["translations"]["app_title"])

    def test_11_onnx_export_endpoint(self):
        res = self.client.post("/api/export-onnx")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertTrue(os.path.exists(data["file"]))

    def test_12_chat_endpoint(self):
        """Verify AI Agronomist chatbot endpoint."""
        # 1. General query
        res1 = self.client.post("/api/chat", json={"message": "What is the best fungicide for tomato late blight?"})
        self.assertEqual(res1.status_code, 200)
        d1 = res1.json()
        self.assertIn("reply", d1)
        self.assertIn("Ridomil Gold", d1["reply"])
        self.assertGreater(len(d1["suggested_actions"]), 0)

        # 2. Contextual diagnosis query
        res2 = self.client.post("/api/chat", json={
            "message": "Can I spray today and what is the dosage?",
            "context": {
                "crop": "Sugarcane",
                "disease": "Sugarcane Red Rot",
                "class_name": "Sugarcane___Red_Rot"
            }
        })
        self.assertEqual(res2.status_code, 200)
        d2 = res2.json()
        self.assertIn("reply", d2)
        self.assertIn("Sugarcane", d2["reply"])

    def test_13_farmer_feedback_and_active_learning(self):
        """Verify farmer feedback submission and active learning enqueueing."""
        fb_payload = {
            "scan_id": "SCN-TEST-001",
            "is_accurate": False,
            "corrected_crop": "Tomato",
            "corrected_disease": "Early Blight",
            "comments": "Target lesions are concentric rings, typical of Alternaria solani"
        }
        res = self.client.post("/api/feedback", json=fb_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["feedback"]["corrected_crop"], "Tomato")

    def test_14_mlops_status_and_queue_management(self):
        """Verify MLOps status, queue inspection, and sample approval."""
        # 1. Get status
        status_res = self.client.get("/api/mlops/status")
        self.assertEqual(status_res.status_code, 200)
        s_data = status_res.json()
        self.assertIn("model_metadata", s_data)
        self.assertIn("queue_statistics", s_data)
        self.assertEqual(s_data["model_metadata"]["classes_supported"], 67)

        # 2. Get queue
        q_res = self.client.get("/api/mlops/queue")
        self.assertEqual(q_res.status_code, 200)
        q_data = q_res.json()
        self.assertIn("samples", q_data)

        # 3. Approve a sample if present
        if len(q_data["samples"]) > 0:
            sample_id = q_data["samples"][0]["sample_id"]
            app_res = self.client.post("/api/mlops/approve-sample", json={
                "sample_id": sample_id,
                "status": "approved_for_training"
            })
            self.assertEqual(app_res.status_code, 200)
            self.assertTrue(app_res.json()["success"])

    def test_15_continuous_retraining_pipeline(self):
        """Verify continuous fine-tuning, validation thresholding, and zero-downtime hot reloading."""
        retrain_res = self.client.post("/api/mlops/trigger-retrain", json={
            "epochs": 2,
            "learning_rate": 0.0001
        })
        self.assertEqual(retrain_res.status_code, 200)
        r_data = retrain_res.json()
        self.assertTrue(r_data["success"])
        self.assertIn("new_version", r_data)
        self.assertEqual(r_data["validation_accuracy"], 100.0)
        self.assertIn("hot_reload", r_data)

    def test_16_spray_window_psychrometrics_and_rainfastness(self):
        """Verify Delta-T psychrometrics, chemical rainfastness database, and spray window REST endpoints."""
        from backend.spray_engine import calculate_delta_t, evaluate_wind_drift_hazard, RAINFASTNESS_DB

        # 1. Delta-T psychrometric computation test
        dt = calculate_delta_t(temp_c=25.0, humidity_pct=60.0)
        self.assertIn("delta_t_c", dt)
        self.assertGreater(dt["delta_t_c"], 4.0)
        self.assertLess(dt["delta_t_c"], 8.0)
        self.assertEqual(dt["rating"], "optimal")

        # 2. Inversion and Drift detection tests
        inversion = evaluate_wind_drift_hazard(wind_speed_kmh=1.5)
        self.assertEqual(inversion["rating"], "caution")
        self.assertIn("Inversion", inversion["status"])

        drift = evaluate_wind_drift_hazard(wind_speed_kmh=26.0)
        self.assertEqual(drift["rating"], "unsuitable")

        # 3. Test /api/spray-chemicals
        chem_res = self.client.get("/api/spray-chemicals")
        self.assertEqual(chem_res.status_code, 200)
        c_data = chem_res.json()
        self.assertIn("chemicals", c_data)
        self.assertGreaterEqual(len(c_data["chemicals"]), 6)

        # 4. Test /api/spray-window endpoint
        spray_res = self.client.get("/api/spray-window?chemical=systemic_fungicide")
        self.assertEqual(spray_res.status_code, 200)
        s_data = spray_res.json()
        self.assertIn("spray_window_analysis", s_data)
        analysis = s_data["spray_window_analysis"]
        self.assertIn("next_safe_window", analysis)
        self.assertIn("hourly_timeline", analysis)
        self.assertEqual(len(analysis["hourly_timeline"]), 48)

    def test_17_hindi_multilingual_chatbot_and_i18n(self):
        """Verify Hindi & multilingual chatbot comprehension and i18n localization dictionary."""
        # 1. Test Hindi chat prompt in Devanagari
        chat_hi = self.client.post("/api/chat", json={
            "message": "टमाटर में झुलसा रोग का इलाज और दवा क्या है?",
            "history": [],
            "language": "hi"
        })
        self.assertEqual(chat_hi.status_code, 200)
        hi_data = chat_hi.json()
        self.assertIn("reply", hi_data)
        self.assertTrue("टमाटर" in hi_data["reply"] or "झुलसा" in hi_data["reply"] or "Tomato" in hi_data["reply"])
        self.assertIn("suggested_actions", hi_data)

        # 2. Test Hindi chat prompt in Hinglish transliteration
        chat_hinglish = self.client.post("/api/chat", json={
            "message": "gehu me peela ratua ki dawa aur khurak batao",
            "history": [],
            "language": "hi"
        })
        self.assertEqual(chat_hinglish.status_code, 200)
        hg_data = chat_hinglish.json()
        self.assertIn("reply", hg_data)

        # 3. Test i18n dictionary for Hindi and English
        i18n_hi = self.client.get("/api/i18n/hi")
        self.assertEqual(i18n_hi.status_code, 200)
        t_hi = i18n_hi.json()["translations"]
        self.assertIn("app_title", t_hi)
        self.assertEqual(t_hi["app_title"], "एग्रो-एआई (AgroAI)")
        self.assertIn("spray_engine_title", t_hi)
        self.assertIn("chatbot_title", t_hi)

        i18n_en = self.client.get("/api/i18n/en")
        self.assertEqual(i18n_en.status_code, 200)
        t_en = i18n_en.json()["translations"]
        self.assertEqual(t_en["app_title"], "AgroAI")

    def test_18_whatsapp_bot_integration(self):
        """Verify WhatsApp Bot webhooks, leaf photo prescriptions, GPS spray advice, and simulation playground."""
        # 1. Config endpoint
        cfg = self.client.get("/api/whatsapp/config")
        self.assertEqual(cfg.status_code, 200)
        c_data = cfg.json()
        self.assertTrue(c_data["bot_active"])
        self.assertIn("deep_link", c_data)
        self.assertIn("features", c_data)

        # 2. Meta Webhook Verification challenge
        meta_ver = self.client.get("/api/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=agroai_whatsapp_token&hub.challenge=CHALLENGE_ACCEPTED_123")
        self.assertEqual(meta_ver.status_code, 200)
        self.assertEqual(meta_ver.text, "CHALLENGE_ACCEPTED_123")

        # 3. WhatsApp Simulate: Text Q&A in Hindi
        sim_hi = self.client.post("/api/whatsapp/simulate", json={
            "message": "टमाटर में झुलसा रोग की दवा",
            "language": "hi"
        })
        self.assertEqual(sim_hi.status_code, 200)
        hi_res = sim_hi.json()
        self.assertIn("reply", hi_res)
        self.assertEqual(hi_res["type"], "chat_advisory")

        # 4. WhatsApp Simulate: Leaf Photo Diagnosis
        sim_photo = self.client.post("/api/whatsapp/simulate", json={
            "sample_id": "tomato_early_blight",
            "language": "en"
        })
        self.assertEqual(sim_photo.status_code, 200)
        p_res = sim_photo.json()
        self.assertEqual(p_res["type"], "diagnosis")
        self.assertIn("Prescription", p_res["reply"])
        self.assertIn("Tomato", p_res["reply"])

        # 5. WhatsApp Simulate: GPS Location Spray Check
        sim_loc = self.client.post("/api/whatsapp/simulate", json={
            "latitude": 30.90,
            "longitude": 75.85,
            "language": "en"
        })
        self.assertEqual(sim_loc.status_code, 200)
        l_res = sim_loc.json()
        self.assertEqual(l_res["type"], "weather_advisory")
        self.assertIn("Delta-T", l_res["reply"])
        self.assertIn("Spray Decision", l_res["reply"])

        # 6. Twilio Form Webhook with TwiML XML response
        twilio_res = self.client.post(
            "/api/whatsapp/webhook",
            data={"From": "whatsapp:+1234567890", "Body": "Help"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        self.assertEqual(twilio_res.status_code, 200)
        self.assertIn("<Response>", twilio_res.text)
        self.assertIn("<Message>", twilio_res.text)

    def test_19_satellite_ndvi_field_mapping(self):
        """Verify Satellite NDVI Field Mapping, multispectral raster generation, zonal stats, VRA prescriptions, and GeoJSON export."""
        # 1. Test sample fields endpoint
        samples_res = self.client.get("/api/ndvi/sample-fields")
        self.assertEqual(samples_res.status_code, 200)
        s_data = samples_res.json()
        self.assertIn("fields", s_data)
        self.assertGreaterEqual(len(s_data["fields"]), 5)
        
        # Verify first field structure (Punjab Wheat)
        f0 = s_data["fields"][0]
        self.assertEqual(f0["crop"], "Wheat")
        self.assertIn("lat", f0)
        self.assertIn("lon", f0)

        # 2. Test field analysis POST endpoint
        analysis_res = self.client.post("/api/ndvi/analyze-field", json={
            "lat": 30.9010,
            "lon": 75.8573,
            "crop": "Wheat",
            "area_hectares": 14.5,
            "field_name": "Ludhiana Precision Wheat Parcel"
        })
        self.assertEqual(analysis_res.status_code, 200)
        a_data = analysis_res.json()
        
        # Verify field metadata
        self.assertIn("field_metadata", a_data)
        self.assertEqual(a_data["field_metadata"]["crop"], "Wheat")
        self.assertEqual(a_data["field_metadata"]["area_hectares"], 14.5)

        # Verify raster grid & statistics
        self.assertIn("raster", a_data)
        raster = a_data["raster"]
        self.assertEqual(raster["grid_size"], 24)
        self.assertEqual(len(raster["cells"]), 24)
        self.assertEqual(len(raster["cells"][0]), 24)
        
        stats = raster["statistics"]
        self.assertIn("mean_ndvi", stats)
        self.assertGreaterEqual(stats["mean_ndvi"], 0.2)
        self.assertLessEqual(stats["mean_ndvi"], 1.0)
        self.assertIn("field_uniformity_score", stats)
        
        # Verify Zonal Breakdown
        zonal = raster["zonal_breakdown"]
        self.assertIn("high_vigor_pct", zonal)
        self.assertIn("moderate_stress_pct", zonal)
        self.assertIn("severe_anomaly_pct", zonal)
        total_pct = zonal["high_vigor_pct"] + zonal["moderate_stress_pct"] + zonal["severe_anomaly_pct"]
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

        # Verify VRA Fertilizer Prescription
        self.assertIn("vra_fertilizer_prescription", a_data)
        vra = a_data["vra_fertilizer_prescription"]
        self.assertIn("zones", vra)
        self.assertEqual(len(vra["zones"]), 3)
        self.assertIn("total_fertilizer_demand", vra)
        self.assertIn("cost_and_input_savings_pct", vra["total_fertilizer_demand"])

        # Verify 180-Day Growth Curve
        self.assertIn("multi_temporal_growth_curve", a_data)
        curve = a_data["multi_temporal_growth_curve"]
        self.assertEqual(len(curve), 6)
        self.assertEqual(curve[0]["stage"], "Emergence & Seedling")

        # 3. Test GeoJSON export endpoint
        geojson_res = self.client.get("/api/ndvi/export-geojson?lat=30.9010&lon=75.8573&crop=Wheat&area=14.5")
        self.assertEqual(geojson_res.status_code, 200)
        g_data = geojson_res.json()
        self.assertEqual(g_data["type"], "FeatureCollection")
        self.assertGreater(len(g_data["features"]), 0)
        self.assertEqual(g_data["features"][0]["geometry"]["type"], "Polygon")

if __name__ == "__main__":
    unittest.main()




