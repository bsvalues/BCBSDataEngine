"""
Test package for the BCBS Values property valuation system.

This package contains test modules for both unit and integration testing.
Key test modules include:

- test_core_valuation.py:         Core valuation algorithm tests
- test_integration_advanced.py:   End-to-end integration tests for the API and ETL pipeline
- test_etl_integration.py:        Focused ETL pipeline integration tests
- test_advanced_valuation.py:     Advanced valuation model tests
- test_enhanced_gis_features.py:  Tests for GIS feature extraction and integration

Tests can be run using pytest:
    pytest -xvs ./tests/

For integration tests only:
    pytest -xvs ./tests/test_integration_advanced.py ./tests/test_etl_integration.py

For core valuation tests only:
    pytest -xvs ./tests/test_core_valuation.py

Author: BCBS Test Engineering Team
Last Updated: 2025-03-31
"""

import os
import sys

# Add the parent directory to the path so that imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Make sure pytest can discover test functions in the modules
__all__ = [
    'test_core_valuation',
    'test_advanced_valuation',
    'test_enhanced_gis_features',
    'test_integration_advanced',
    'test_etl_integration',
    'test_valuation',
    'test_integration',
    'test_data_validation',
    'test_mls_scraper',
    'test_narrpr_scraper',
    'test_pacs_import'
]