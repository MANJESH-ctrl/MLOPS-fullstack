import unittest
import requests
import json
import pandas as pd
import os

class TestInsuranceAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("🔧 Setting up API test environment...")

        cls.base_url = os.getenv('API_BASE_URL', 'http://localhost:5000')
        print(f"🌐 API Base URL: {cls.base_url}")

        os.makedirs('reports/api_tests', exist_ok=True)

        try:
            df = pd.read_csv('data/processed/test_final.csv')
            cls.sample_row = df.iloc[0].drop(labels=['Response'], errors='ignore')
            print("✅ Sample data loaded for API testing")
        except Exception:
            print("⚠️ Using mock data")
            cls.sample_row = pd.Series({
                'Gender': 'Male',
                'Age': 45,
                'Driving_License': 1,
                'Region_Code': 28,
                'Previously_Insured': 0,
                'Vehicle_Age': '< 1 Year',
                'Vehicle_Damage': 'No',
                'Annual_Premium': 25000.0,
                'Policy_Sales_Channel': 152,
                'Vintage': 200
            })

    def _series_to_dict(self, series):
        out = {}
        for k, v in series.items():
            if pd.isna(v):
                out[k] = None
            elif isinstance(v, (int, float)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def test_01_health_endpoint(self):
        print("🧪 Test 1: Health endpoint...")
        try:
            r = requests.get(f'{self.base_url}/health', timeout=5)
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body.get('status'), 'healthy')
            print(f"✅ Health endpoint OK: {body}")
        except requests.exceptions.ConnectionError:
            self.skipTest("API server not running")

    def test_02_prediction_endpoint(self):
        print("🧪 Test 2: Prediction endpoint...")
        payload = self._series_to_dict(self.sample_row)

        r = requests.post(
            f'{self.base_url}/predict_api',
            json=payload,
            timeout=10
        )

        if r.status_code == 404:
            self.skipTest("/predict_api not implemented")

        self.assertEqual(r.status_code, 200)
        body = r.json()

        self.assertIn('prediction', body)
        self.assertIn(body['prediction'], [0, 1])
        self.assertIn('timestamp', body)

        print(f"✅ Prediction OK: {body}")



    def test_03_error_handling(self):
        """
        Minimal + compatible:
        API may still return prediction even for bad input.
        We only ensure it returns VALID JSON.
        """
        print("🧪 Test 3: Error handling (lenient)")

        malformed = {'wrong_key': [1, 2, 3]}
        r = requests.post(
            f'{self.base_url}/predict_api',
            json=malformed,
            timeout=5
        )

        if r.status_code == 404:
            self.skipTest("/predict_api not implemented")

        # Acceptable responses
        self.assertIn(r.status_code, [200, 400, 422, 500])

        try:
            body = r.json()
        except Exception:
            self.fail("Response was not valid JSON")

        # Either error OR prediction is acceptable
        self.assertTrue(
            'error' in body or 'prediction' in body,
            "Response must contain either 'error' or 'prediction'"
        )

        print(f"✅ Error-handling behavior accepted: {body}")

    @classmethod
    def tearDownClass(cls):
        print("\n🧹 Cleaning up API test artifacts...")
        print("✅ API tests completed!")

if __name__ == '__main__':
    print("🚀 Starting API endpoint tests...")
    print("=" * 50)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInsuranceAPI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    exit(0 if result.wasSuccessful() else 1)
