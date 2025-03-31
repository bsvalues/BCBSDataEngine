"""
BCBS Values - GIS Integration Module

This module provides Geographic Information System (GIS) integration
for the property valuation system, allowing spatial analysis to be
incorporated into property valuations.
"""

import logging
import random
import math
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# School district data (simulated)
SCHOOL_DISTRICTS = {
    'seattle': {
        'boundaries': {
            'north': 47.7340,
            'south': 47.5000,
            'east': -122.2300,
            'west': -122.4400
        },
        'elementary_districts': [
            {
                'name': 'North Seattle Elementary District',
                'rating': 8.5,
                'boundaries': {
                    'north': 47.7340,
                    'south': 47.6800,
                    'east': -122.2800,
                    'west': -122.4200
                }
            },
            {
                'name': 'Central Seattle Elementary District',
                'rating': 7.8,
                'boundaries': {
                    'north': 47.6800,
                    'south': 47.6000,
                    'east': -122.2500,
                    'west': -122.3800
                }
            },
            {
                'name': 'South Seattle Elementary District',
                'rating': 6.9,
                'boundaries': {
                    'north': 47.6000,
                    'south': 47.5000,
                    'east': -122.2300,
                    'west': -122.4400
                }
            }
        ]
    }
}

# Neighborhood quality data (simulated)
NEIGHBORHOOD_QUALITY = {
    'downtown': {
        'walkability': 95,
        'transit_score': 92,
        'noise_level': 75,
        'crime_index': 65,
        'restaurant_density': 95,
        'park_access': 80,
        'amenities_score': 90
    },
    'queen anne': {
        'walkability': 88,
        'transit_score': 78,
        'noise_level': 55,
        'crime_index': 35,
        'restaurant_density': 85,
        'park_access': 75,
        'amenities_score': 82
    },
    'ballard': {
        'walkability': 85,
        'transit_score': 72,
        'noise_level': 60,
        'crime_index': 40,
        'restaurant_density': 88,
        'park_access': 78,
        'amenities_score': 80
    },
    'fremont': {
        'walkability': 82,
        'transit_score': 68,
        'noise_level': 58,
        'crime_index': 38,
        'restaurant_density': 85,
        'park_access': 80,
        'amenities_score': 78
    },
    'capitol hill': {
        'walkability': 92,
        'transit_score': 85,
        'noise_level': 72,
        'crime_index': 55,
        'restaurant_density': 95,
        'park_access': 78,
        'amenities_score': 90
    },
    'west seattle': {
        'walkability': 68,
        'transit_score': 62,
        'noise_level': 50,
        'crime_index': 35,
        'restaurant_density': 72,
        'park_access': 85,
        'amenities_score': 70
    },
    'beacon hill': {
        'walkability': 65,
        'transit_score': 70,
        'noise_level': 55,
        'crime_index': 45,
        'restaurant_density': 60,
        'park_access': 70,
        'amenities_score': 65
    },
    'rainier valley': {
        'walkability': 60,
        'transit_score': 65,
        'noise_level': 60,
        'crime_index': 60,
        'restaurant_density': 55,
        'park_access': 65,
        'amenities_score': 60
    },
    'university district': {
        'walkability': 90,
        'transit_score': 85,
        'noise_level': 70,
        'crime_index': 50,
        'restaurant_density': 90,
        'park_access': 75,
        'amenities_score': 85
    },
    'northgate': {
        'walkability': 65,
        'transit_score': 75,
        'noise_level': 65,
        'crime_index': 45,
        'restaurant_density': 70,
        'park_access': 60,
        'amenities_score': 65
    }
}

# Flood risk zones (simulated)
FLOOD_ZONES = [
    {
        'name': 'Downtown Waterfront',
        'risk_level': 'Moderate',
        'risk_factor': 3.5,
        'boundaries': {
            'north': 47.6150,
            'south': 47.5950,
            'east': -122.3300,
            'west': -122.3500
        }
    },
    {
        'name': 'South Seattle Lowlands',
        'risk_level': 'High',
        'risk_factor': 7.2,
        'boundaries': {
            'north': 47.5500,
            'south': 47.5100,
            'east': -122.2800,
            'west': -122.3200
        }
    },
    {
        'name': 'Duwamish River Basin',
        'risk_level': 'Very High',
        'risk_factor': 8.5,
        'boundaries': {
            'north': 47.5700,
            'south': 47.5300,
            'east': -122.3000,
            'west': -122.3400
        }
    }
]


def get_location_score(latitude, longitude):
    """
    Calculate a location score based on GIS data for the given coordinates.
    
    This score combines multiple geographic factors including walkability,
    proximity to amenities, and neighborhood quality.
    
    Args:
        latitude: Decimal latitude coordinate
        longitude: Decimal longitude coordinate
        
    Returns:
        dict: Location score data including overall score and contributing factors
    """
    try:
        # Create a seed from the coordinates for deterministic random values
        coord_seed = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(coord_seed)
        
        # Determine the neighborhood (in a real system, this would use geospatial queries)
        neighborhood = _determine_neighborhood(latitude, longitude)
        
        # Get neighborhood quality data if available, otherwise use defaults
        if neighborhood and neighborhood.lower() in NEIGHBORHOOD_QUALITY:
            quality_data = NEIGHBORHOOD_QUALITY[neighborhood.lower()]
        else:
            # Default neighborhood quality metrics if not found
            quality_data = {
                'walkability': random.randint(50, 85),
                'transit_score': random.randint(40, 80),
                'noise_level': random.randint(40, 70),
                'crime_index': random.randint(30, 60),
                'restaurant_density': random.randint(50, 80),
                'park_access': random.randint(50, 80),
                'amenities_score': random.randint(50, 80)
            }
        
        # Calculate a location score (0-100) based on the quality metrics
        # Higher walkability, transit, restaurants, parks and amenities are better
        # Lower noise and crime are better
        positive_factors = (
            quality_data['walkability'] +
            quality_data['transit_score'] +
            quality_data['restaurant_density'] +
            quality_data['park_access'] +
            quality_data['amenities_score']
        ) / 5
        
        negative_factors = (
            quality_data['noise_level'] +
            quality_data['crime_index']
        ) / 2
        
        # Invert negative factors (100 - value) so higher is better
        negative_contribution = (100 - negative_factors) * 0.3  # 30% weight to negative factors
        positive_contribution = positive_factors * 0.7  # 70% weight to positive factors
        
        # Combine factors for a score between 0-100
        location_score = round(negative_contribution + positive_contribution)
        
        # Add a small random factor (+/- 5) for natural variation
        # This simulates other unmeasured factors
        location_score = min(100, max(0, location_score + random.randint(-5, 5)))
        
        # Generate detailed factor data
        factors = {
            'neighborhood': neighborhood,
            'walkability': quality_data['walkability'],
            'transit_access': quality_data['transit_score'],
            'noise_level': quality_data['noise_level'],
            'crime_level': quality_data['crime_index'],
            'restaurant_access': quality_data['restaurant_density'],
            'park_access': quality_data['park_access'],
            'nearby_amenities': quality_data['amenities_score'],
            'positive_contribution': round(positive_contribution, 1),
            'negative_contribution': round(negative_contribution, 1)
        }
        
        # Reset random seed
        random.seed()
        
        return {
            'score': location_score,
            'factors': factors,
            'grade': _score_to_grade(location_score)
        }
        
    except Exception as e:
        logger.error(f"Error calculating location score: {e}")
        # Return a default score if there's an error
        return {
            'score': 50,
            'factors': {'error': str(e)},
            'grade': 'C'
        }


def get_school_district_info(latitude, longitude):
    """
    Get school district information for the given coordinates.
    
    Args:
        latitude: Decimal latitude coordinate
        longitude: Decimal longitude coordinate
        
    Returns:
        dict: School district information and ratings
    """
    try:
        # Create a seed from the coordinates for deterministic random values
        coord_seed = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(coord_seed)
        
        # Check if the coordinates are within a known school district
        district_info = _find_school_district(latitude, longitude)
        
        if district_info:
            # Add some simulated metrics about the district
            district_info.update({
                'student_teacher_ratio': round(random.uniform(15, 25), 1),
                'graduation_rate': round(random.uniform(75, 98), 1),
                'college_acceptance_rate': round(random.uniform(60, 95), 1),
                'test_scores_percentile': round(random.uniform(40, 95))
            })
        else:
            # If no specific district was found, generate generic school info
            elementary_rating = round(random.uniform(5.0, 9.0), 1)
            middle_school_rating = round(random.uniform(4.5, 8.5), 1)
            high_school_rating = round(random.uniform(4.0, 9.0), 1)
            
            # Calculate an overall rating weighted slightly toward elementary
            overall_rating = round((elementary_rating * 0.4 + 
                                  middle_school_rating * 0.3 + 
                                  high_school_rating * 0.3), 1)
            
            district_info = {
                'district_name': 'Seattle School District',
                'elementary_school': {
                    'name': 'Local Elementary School',
                    'rating': elementary_rating,
                    'distance_miles': round(random.uniform(0.2, 1.5), 1)
                },
                'middle_school': {
                    'name': 'Local Middle School',
                    'rating': middle_school_rating,
                    'distance_miles': round(random.uniform(0.5, 2.5), 1)
                },
                'high_school': {
                    'name': 'Local High School',
                    'rating': high_school_rating,
                    'distance_miles': round(random.uniform(0.8, 3.5), 1)
                },
                'overall_rating': overall_rating,
                'student_teacher_ratio': round(random.uniform(15, 25), 1),
                'graduation_rate': round(random.uniform(75, 98), 1),
                'college_acceptance_rate': round(random.uniform(60, 95), 1),
                'test_scores_percentile': round(random.uniform(40, 95))
            }
        
        # Reset random seed
        random.seed()
        
        return district_info
        
    except Exception as e:
        logger.error(f"Error retrieving school district info: {e}")
        # Return a default response if there's an error
        return {
            'district_name': 'Unknown District',
            'overall_rating': 5.0,
            'error': str(e)
        }


def get_flood_risk_assessment(latitude, longitude):
    """
    Assess flood risk for the given coordinates.
    
    Args:
        latitude: Decimal latitude coordinate
        longitude: Decimal longitude coordinate
        
    Returns:
        dict: Flood risk assessment information
    """
    try:
        # Check if the coordinates are within a known flood zone
        for zone in FLOOD_ZONES:
            boundaries = zone['boundaries']
            if (boundaries['north'] >= latitude >= boundaries['south'] and
                boundaries['east'] >= longitude >= boundaries['west']):
                # Found a flood zone match
                return {
                    'risk_level': zone['risk_level'],
                    'risk_factor': zone['risk_factor'],
                    'zone_name': zone['name'],
                    'insurancwe_recommended': zone['risk_factor'] > 5.0,
                    'annual_flood_probability': _risk_factor_to_probability(zone['risk_factor']),
                    'last_significant_flood': f"{random.randint(1990, 2019)}-{random.randint(1, 12):02d}"
                }
        
        # If not in a known flood zone, create a simulated low risk assessment
        # Use coordinate-based seed for consistency
        coord_seed = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(coord_seed)
        
        # Most areas have low flood risk
        risk_factor = random.uniform(1.0, 3.0) 
        
        # Special case - properties close to water bodies might have higher risk
        # This would use real GIS data in a production system
        if ((47.65 > latitude > 47.55 and -122.32 > longitude > -122.35) or  # Lake Union area
            (47.58 > latitude > 47.50 and -122.27 > longitude > -122.33)):  # Lake Washington area
            risk_factor = random.uniform(2.5, 4.5)
        
        result = {
            'risk_level': _risk_factor_to_level(risk_factor),
            'risk_factor': round(risk_factor, 1),
            'zone_name': 'General Assessment Area',
            'insurance_recommended': risk_factor > 5.0,
            'annual_flood_probability': _risk_factor_to_probability(risk_factor),
            'historical_flooding': risk_factor > 3.0
        }
        
        # Reset random seed
        random.seed()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in flood risk assessment: {e}")
        # Return a default low risk if there's an error
        return {
            'risk_level': 'Low',
            'risk_factor': 2.0,
            'zone_name': 'Error Assessment',
            'error': str(e)
        }


def _determine_neighborhood(latitude, longitude):
    """
    Determine the neighborhood based on coordinates.
    
    This is a simplified version that would be replaced with actual
    GIS boundary queries in a production system.
    """
    # Create a deterministic but seemingly random mapping of coordinates to neighborhoods
    # In a real system, this would use actual GIS data with neighborhood boundaries
    
    # Downtown Seattle approximate boundaries
    if 47.62 > latitude > 47.59 and -122.32 > longitude > -122.35:
        return "Downtown"
        
    # Queen Anne approximate boundaries
    elif 47.65 > latitude > 47.62 and -122.34 > longitude > -122.37:
        return "Queen Anne"
        
    # Capitol Hill approximate boundaries
    elif 47.63 > latitude > 47.60 and -122.31 > longitude > -122.34:
        return "Capitol Hill"
    
    # Use a hash of the coordinates for a deterministic but pseudo-random assignment
    # for regions we don't explicitly map
    coord_seed = abs(hash(f"{latitude}{longitude}")) % 1000000
    random.seed(coord_seed)
    
    neighborhoods = list(NEIGHBORHOOD_QUALITY.keys())
    selected = neighborhoods[coord_seed % len(neighborhoods)]
    
    # Reset random seed
    random.seed()
    
    # Return with proper capitalization
    return selected.title()


def _find_school_district(latitude, longitude):
    """Find the school district that contains the given coordinates."""
    # Check if coordinates are in the Seattle district
    seattle = SCHOOL_DISTRICTS['seattle']
    boundaries = seattle['boundaries']
    
    if not (boundaries['north'] >= latitude >= boundaries['south'] and
            boundaries['east'] >= longitude >= boundaries['west']):
        # Not in Seattle boundaries
        return None
    
    # Check elementary districts
    elementary_district = None
    for district in seattle['elementary_districts']:
        district_bounds = district['boundaries']
        if (district_bounds['north'] >= latitude >= district_bounds['south'] and
            district_bounds['east'] >= longitude >= district_bounds['west']):
            elementary_district = district
            break
    
    # Create a seed from the coordinates for deterministic random values
    coord_seed = abs(hash(f"{latitude}{longitude}")) % 1000000
    random.seed(coord_seed)
    
    # If no specific elementary district was found, use the overall Seattle district
    if not elementary_district:
        elementary_rating = round(random.uniform(5.0, 9.0), 1)
        elementary_district = {
            'name': 'Seattle Elementary District',
            'rating': elementary_rating
        }
    
    # Generate ratings for middle and high schools
    middle_school_rating = min(10.0, max(1.0, elementary_district['rating'] + random.uniform(-1.0, 1.0)))
    high_school_rating = min(10.0, max(1.0, middle_school_rating + random.uniform(-1.0, 1.0)))
    
    # Calculate an overall rating weighted slightly toward elementary
    overall_rating = round((elementary_district['rating'] * 0.4 + 
                          middle_school_rating * 0.3 + 
                          high_school_rating * 0.3), 1)
    
    # Reset random seed
    random.seed()
    
    return {
        'district_name': 'Seattle School District',
        'elementary_school': {
            'name': elementary_district['name'],
            'rating': elementary_district['rating'],
            'distance_miles': round(random.uniform(0.2, 1.5), 1)
        },
        'middle_school': {
            'name': 'Seattle Middle School',
            'rating': middle_school_rating,
            'distance_miles': round(random.uniform(0.5, 2.5), 1)
        },
        'high_school': {
            'name': 'Seattle High School',
            'rating': high_school_rating,
            'distance_miles': round(random.uniform(0.8, 3.5), 1)
        },
        'overall_rating': overall_rating
    }


def _score_to_grade(score):
    """Convert a numeric score (0-100) to a letter grade (A+ to F)."""
    if score >= 97:
        return 'A+'
    elif score >= 93:
        return 'A'
    elif score >= 90:
        return 'A-'
    elif score >= 87:
        return 'B+'
    elif score >= 83:
        return 'B'
    elif score >= 80:
        return 'B-'
    elif score >= 77:
        return 'C+'
    elif score >= 73:
        return 'C'
    elif score >= 70:
        return 'C-'
    elif score >= 67:
        return 'D+'
    elif score >= 63:
        return 'D'
    elif score >= 60:
        return 'D-'
    else:
        return 'F'


def _risk_factor_to_level(risk_factor):
    """Convert a numeric risk factor to a descriptive risk level."""
    if risk_factor < 2.0:
        return 'Very Low'
    elif risk_factor < 4.0:
        return 'Low'
    elif risk_factor < 6.0:
        return 'Moderate'
    elif risk_factor < 8.0:
        return 'High'
    else:
        return 'Very High'


def _risk_factor_to_probability(risk_factor):
    """Convert a risk factor (1-10) to an annual flood probability percentage."""
    # Exponential relationship between risk factor and probability
    # A risk factor of 1 = 0.1% annual probability
    # A risk factor of 10 = 10% annual probability
    return round(0.1 * math.exp((risk_factor - 1) * 0.35), 2)