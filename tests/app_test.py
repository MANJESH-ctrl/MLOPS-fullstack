import unittest
import json
from flask_app.app import app # Adjust import based on your app structure

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<html", response.data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")  # Adjusted to match MINE's expected 'healthy'

    def test_predict_api_success(self):
        payload = [
            {
                "Gender": "Male",
                "Age": 45,
                "Driving_License": 1,
                "Region_Code": 28,
                "Previously_Insured": 0,
                "Vehicle_Age": "< 1 Year",
                "Vehicle_Damage": "No",
                "Annual_Premium": 25000.0,
                "Policy_Sales_Channel": 152,
                "Vintage": 200
            }
        ]
        response = self.client.post(
            "/predict_api",  # Adjusted to match MINE's endpoint
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], [0, 1])  # Adjusted to match MINE's binary prediction check
        self.assertIn("timestamp", data)  # Kept from MINE

    def test_predict_api_missing_features(self):
        payload = [
            {
                "Gender": "Male",
                "Age": 45
                # missing remaining features
            }
        ]
        response = self.client.post(
            "/predict_api",  # Adjusted to match MINE's endpoint
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_predict_form_success(self):
        form_data = {
            "Gender": "Male",
            "Age": "45",
            "Driving_License": "1",
            "Region_Code": "28",
            "Previously_Insured": "0",
            "Vehicle_Age": "< 1 Year",
            "Vehicle_Damage": "No",
            "Annual_Premium": "25000.0",
            "Policy_Sales_Channel": "152",
            "Vintage": "200"
        }
        response = self.client.post("/predict-form", data=form_data)  # Assuming this endpoint exists; adjust if needed
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predicted", response.data)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"app_request_count", response.data)

if __name__ == "__main__":
    unittest.main(verbosity=2)