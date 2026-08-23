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

if __name__ == "__main__":
    unittest.main()


