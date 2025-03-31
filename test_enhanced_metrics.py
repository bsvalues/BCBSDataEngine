"""
Unit tests for the enhanced model metrics and comparison functionality.
"""

import unittest
import pandas as pd
import numpy as np
from src.valuation import AdvancedValuationEngine

class TestEnhancedMetrics(unittest.TestCase):
    """Tests for the enhanced metrics and model comparison functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = AdvancedValuationEngine()
    
    def test_model_strengths_extraction(self):
        """Test the _extract_model_strengths method."""
        # Test with linear model info
        linear_info = {
            'model_type': 'linear',
            'r_squared': 0.83,
            'rmse': 35000,
            'model_notes': ['Good performance on test data']
        }
        
        strengths = self.engine._extract_model_strengths(linear_info)
        
        # Verify strengths contains expected keys and values
        self.assertIn('interpretability', strengths)
        self.assertIn('feature_impact', strengths)
        self.assertIn('prediction_speed', strengths)
        self.assertIn('best_for', strengths)
        self.assertEqual(strengths['interpretability'], 'High - Coefficients directly show feature impact')
        
        # Test with lightgbm model info including overfitting metrics
        lightgbm_info = {
            'model_type': 'lightgbm',
            'r_squared': 0.89,
            'rmse': 28000,
            'model_notes': ['Handles non-linear relationships well'],
            'overfitting_metrics': {
                'r_squared_diff': 0.15,
                'train_r_squared': 0.95
            }
        }
        
        strengths = self.engine._extract_model_strengths(lightgbm_info)
        
        # Verify strengths contains expected keys and overfitting assessment
        self.assertIn('overfitting_assessment', strengths)
        self.assertEqual(strengths['overfitting_assessment'], 'Medium risk - Some gap between training and test performance')
        self.assertEqual(strengths['best_for'], 'Large datasets with complex non-linear relationships and interactions')
    
    def test_model_comparison_summary(self):
        """Test the _generate_model_comparison_summary method."""
        # Create mock model comparison data
        models_compared = {
            'linear': {
                'r_squared': 0.82,
                'rmse': 42000,
                'training_time': 0.15,
                'prediction_time': 0.002
            },
            'ridge': {
                'r_squared': 0.83,
                'rmse': 40000,
                'training_time': 0.18,
                'prediction_time': 0.002
            },
            'lightgbm': {
                'r_squared': 0.88,
                'rmse': 32000,
                'training_time': 1.25,
                'prediction_time': 0.01
            }
        }
        
        summary = self.engine._generate_model_comparison_summary(models_compared)
        
        # Verify summary contains expected keys and values
        self.assertIn('overview', summary)
        self.assertIn('best_model_by_r2', summary)
        self.assertIn('fastest_model', summary)
        self.assertIn('comparison_table', summary)
        
        # Check that lightgbm is identified as the best model by R²
        self.assertIn('lightgbm', summary['best_model_by_r2'])
        
        # Check that linear is identified as the fastest model
        self.assertIn('linear', summary['fastest_model'])
        
        # Check the comparison table has all models
        self.assertEqual(len(summary['comparison_table']), 3)
        
        # Test with only one model (should indicate insufficient models)
        single_model = {'linear': models_compared['linear']}
        summary = self.engine._generate_model_comparison_summary(single_model)
        
        self.assertIn('summary', summary)
        self.assertEqual(summary['summary'], 'Insufficient models to generate comparison')

if __name__ == '__main__':
    unittest.main()