"""
Core valuation functionality for BCBS Values application.

This module provides the core algorithms for calculating property valuations
and saving them to the database.
"""
import logging
import random
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from app import db
from models import Valuation

# Configure logging
logger = logging.getLogger(__name__)

# Constants for valuation calculations
BASE_PRICE_PER_SQFT = {
    'single_family': 250,
    'condo': 325,
    'townhouse': 275,
    'multi_family': 200,
    'land': 50,
    'commercial': 175
}

NEIGHBORHOOD_MULTIPLIERS = {
    # These would ideally come from a database or API
    'downtown': 1.5,
    'capitol hill': 1.3,
    'ballard': 1.2,
    'fremont': 1.25,
    'queen anne': 1.35,
    'west seattle': 1.1,
    'beacon hill': 0.9,
    'rainier valley': 0.85,
    'university district': 1.15,
    'northgate': 0.95,
    'test neighborhood': 1.0  # For testing
}

PROPERTY_AGE_FACTORS = {
    # New construction premium
    (0, 5): 1.1,
    # Newer homes
    (6, 10): 1.05,
    # Established homes
    (11, 20): 1.0,
    # Older homes
    (21, 40): 0.95,
    # Much older homes
    (41, 75): 0.9,
    # Historic homes (may have premium again)
    (76, 100): 0.95,
    # Very old homes
    (101, 1000): 0.85
}


def calculate_basic_valuation(property_obj):
    """
    Calculate a basic property valuation based on property characteristics.
    
    This function implements a simple valuation model that considers:
    - Property type and square footage
    - Neighborhood factors
    - Property age
    - Number of bedrooms and bathrooms
    - Lot size
    
    Args:
        property_obj: A Property object containing details about the property
        
    Returns:
        dict: A dictionary containing the estimated value and confidence score
    """
    logger.info(f"Calculating basic valuation for property: {property_obj.address}")
    
    try:
        # Check if we have the minimum required data
        if not property_obj.property_type:
            property_obj.property_type = 'single_family'  # Default
            logger.warning(f"Property type missing for {property_obj.address}, using default")
        
        # Calculate base price based on property type and square footage
        price_per_sqft = BASE_PRICE_PER_SQFT.get(property_obj.property_type, 200)
        square_feet = property_obj.square_feet or 1500  # Default to 1500 if missing
        
        # Start with basic calculation
        base_value = price_per_sqft * square_feet
        
        # Track confidence reductions due to missing data
        confidence_penalty = 0
        
        # Apply neighborhood factor
        if property_obj.neighborhood:
            neighborhood_lower = property_obj.neighborhood.lower()
            neighborhood_factor = NEIGHBORHOOD_MULTIPLIERS.get(neighborhood_lower, 1.0)
        else:
            neighborhood_factor = 1.0
            confidence_penalty += 0.1
            logger.warning(f"Neighborhood missing for {property_obj.address}")
        
        # Apply property age factor
        current_year = datetime.now().year
        if property_obj.year_built:
            property_age = current_year - property_obj.year_built
            age_factor = 1.0
            
            for age_range, factor in PROPERTY_AGE_FACTORS.items():
                if age_range[0] <= property_age <= age_range[1]:
                    age_factor = factor
                    break
        else:
            age_factor = 1.0
            confidence_penalty += 0.05
            logger.warning(f"Year built missing for {property_obj.address}")
        
        # Apply bedroom/bathroom factors
        if property_obj.bedrooms:
            bedroom_factor = 1.0 + (0.05 * (property_obj.bedrooms - 3))  # 3 is baseline
        else:
            bedroom_factor = 1.0
            confidence_penalty += 0.03
        
        if property_obj.bathrooms:
            bathroom_factor = 1.0 + (0.05 * (property_obj.bathrooms - 2))  # 2 is baseline
        else:
            bathroom_factor = 1.0
            confidence_penalty += 0.03
        
        # Apply lot size factor
        if property_obj.lot_size:
            # Assume baseline is 0.25 acres, adjust by 10% per 0.25 acre difference
            lot_size_factor = 1.0 + (0.1 * (property_obj.lot_size - 0.25) / 0.25)
            # Limit the impact of extremely large lots
            lot_size_factor = max(0.8, min(1.5, lot_size_factor))
        else:
            lot_size_factor = 1.0
            confidence_penalty += 0.03
        
        # Calculate the final value
        final_value = base_value * neighborhood_factor * age_factor * bedroom_factor * bathroom_factor * lot_size_factor
        
        # Apply a small random adjustment to simulate market variance (±3%)
        # This is to model the inherent uncertainty in real estate valuation
        variance_factor = random.uniform(0.97, 1.03)
        final_value = final_value * variance_factor
        
        # Round to nearest $1000
        final_value = round(final_value / 1000) * 1000
        
        # Base confidence score adjusted for missing data
        confidence_score = max(0.1, 0.9 - confidence_penalty)
        
        # Return the valuation results
        return {
            'estimated_value': final_value,
            'confidence_score': confidence_score,
            'valuation_method': 'basic',
            'valuation_date': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in basic valuation calculation: {e}")
        # Provide a fallback valuation with low confidence
        return {
            'estimated_value': 300000,  # Conservative fallback estimate
            'confidence_score': 0.3,    # Low confidence due to error
            'valuation_method': 'basic_fallback',
            'valuation_date': datetime.now(),
            'error': str(e)
        }


def save_valuation(property_id, valuation_result):
    """
    Save a property valuation to the database.
    
    Args:
        property_id: The ID of the property being valued
        valuation_result: The valuation result dictionary
        
    Returns:
        int: The ID of the newly created valuation record
    """
    try:
        # Create a new Valuation record
        valuation = Valuation(
            property_id=property_id,
            estimated_value=valuation_result['estimated_value'],
            confidence_score=valuation_result['confidence_score'],
            valuation_method=valuation_result.get('valuation_method', 'basic'),
            valuation_date=valuation_result.get('valuation_date', datetime.now())
        )
        
        # Add to database and commit
        db.session.add(valuation)
        db.session.commit()
        
        logger.info(f"Saved valuation {valuation.id} for property {property_id}")
        return valuation.id
        
    except Exception as e:
        logger.error(f"Error saving valuation for property {property_id}: {e}")
        db.session.rollback()
        raise