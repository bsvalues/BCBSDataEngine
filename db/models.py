"""
Database models for the property valuation system.
"""
import os
import sys

# Import models from the root models.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import Property, ValidationResult, PropertyValuation

# This file is now just a proxy to the main models.py
# to avoid duplicate table definitions
