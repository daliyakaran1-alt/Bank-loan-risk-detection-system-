"""
Tests for KPI Engine.
"""

import unittest
import pandas as pd
from src.analytics.kpi_engine import KPIEngine

class TestKPIEngine(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 0],
            "AMT_CREDIT": [100000, 200000, 150000, 50000],
            "AMT_INCOME_TOTAL": [50000, 80000, 60000, 40000],
            "CODE_GENDER": ['M', 'F', 'M', 'F'],
            "NAME_EDUCATION_TYPE": ['Secondary', 'Higher education', 'Secondary', 'Secondary']
        })

    def test_calculate_all_kpis(self):
        engine = KPIEngine(target_col="TARGET")
        kpis = engine.calculate_all_kpis(self.sample_df)
        
        self.assertIn("total_applications", kpis)
        self.assertEqual(kpis["total_applications"].value, 4)
        
        self.assertIn("default_rate", kpis)
        self.assertEqual(kpis["default_rate"].value, 0.25)
        
        self.assertIn("avg_credit_amount", kpis)
        self.assertEqual(kpis["avg_credit_amount"].value, 125000.0)
        
        self.assertIn("avg_income", kpis)
        self.assertEqual(kpis["avg_income"].value, 57500.0)

if __name__ == '__main__':
    unittest.main()
