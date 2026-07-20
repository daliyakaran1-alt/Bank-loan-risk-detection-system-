"""
Tests for Analytics Manager.
"""

import unittest
import pandas as pd
from src.analytics.analytics_manager import AnalyticsManager
from src.analytics.analytics_models import AnalyticsResult

class TestAnalyticsManager(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 0],
            "AMT_CREDIT": [100000, 200000, 150000, 50000],
            "AMT_INCOME_TOTAL": [50000, 80000, 60000, 40000],
            "AMT_ANNUITY": [5000, 8000, 6000, 4000],
            "CODE_GENDER": ['M', 'F', 'M', 'F'],
            "NAME_EDUCATION_TYPE": ['Secondary', 'Higher education', 'Secondary', 'Secondary'],
            "OCCUPATION_TYPE": ['Laborers', 'Core staff', 'Laborers', 'Core staff'],
            "NAME_FAMILY_STATUS": ['Married', 'Single', 'Married', 'Married'],
            "DAYS_BIRTH": [-10000, -15000, -12000, -11000]
        })

    def test_analytics_manager_run(self):
        manager = AnalyticsManager(target_col="TARGET")
        result = manager.run_analytics(self.sample_df, dataset_name="Test Dataset")
        
        self.assertIsInstance(result, AnalyticsResult)
        self.assertEqual(result.dataset_name, "Test Dataset")
        self.assertIn("Executive Dashboard", result.sections)
        
        self.assertIsNotNone(result.executive_metrics)
        self.assertEqual(result.executive_metrics.get('total_loan_applications', 0), 4)
        self.assertEqual(result.executive_metrics.get('total_customers', 0), 4)

if __name__ == '__main__':
    unittest.main()
