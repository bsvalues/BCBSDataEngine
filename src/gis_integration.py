"""
BCBS Values - GIS Integration Module

This module provides integration with Geographic Information Systems (GIS)
for obtaining spatial data and location-based insights for property valuation.
"""

import logging
import random
import math
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Constants for GIS integration
# Note: In a real implementation, these would be API endpoints or database connections
GIS_DATA_SOURCES = {
    'school_districts': 'gis_data/school_districts.json',
    'flood_zones': 'gis_data/flood_zones.json',
    'crime_statistics': 'gis_data/crime_statistics.json',
    'transit_routes': 'gis_data/transit_routes.json',
    'property_values': 'gis_data/property_values.json',
    'amenities': 'gis_data/amenities.json',
    'zoning': 'gis_data/zoning.json'
}

# Mock feature weights for location score calculation
FEATURE_WEIGHTS = {
    'school_quality': 0.20,
    'crime_rate': -0.15,
    'flood_risk': -0.10,
    'transit_access': 0.12,
    'amenities_proximity': 0.18,
    'parks_proximity': 0.10,
    'noise_level': -0.05,
    'walkability': 0.10
}

def get_location_score(latitude, longitude, feature_weights=None):
    """
    Calculate a location score based on various GIS features.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        feature_weights: Optional custom weights for various GIS features
        
    Returns:
        dict: Location score and contributing factors
    """
    logger.info(f"Calculating location score for coordinates: {latitude}, {longitude}")
    
    if not latitude or not longitude:
        logger.warning("Missing coordinates for location score calculation")
        return {
            'score': None,
            'message': "Location score calculation requires valid coordinates",
            'factors': {}
        }
    
    # In a real implementation, we would query GIS data sources here
    # For this demo, we'll generate randomized but realistic values
    try:
        # Use weights from parameter if provided, otherwise use defaults
        weights = feature_weights or FEATURE_WEIGHTS
        
        # Generate random scores for each feature (would normally come from GIS data)
        # We're using the coordinates as a seed for consistent results
        seed_value = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(seed_value)
        
        factors = {}
        
        # School quality (higher is better, 1-10 scale)
        factors['school_quality'] = min(10, max(1, random.uniform(5.0, 9.0)))
        
        # Crime rate (lower is better, 1-10 scale)
        factors['crime_rate'] = min(10, max(1, random.uniform(2.0, 7.0)))
        
        # Flood risk (lower is better, 1-10 scale)
        factors['flood_risk'] = min(10, max(1, random.uniform(1.0, 5.0)))
        
        # Transit access (higher is better, 1-10 scale)
        factors['transit_access'] = min(10, max(1, random.uniform(3.0, 8.0)))
        
        # Amenities proximity (higher is better, 1-10 scale)
        factors['amenities_proximity'] = min(10, max(1, random.uniform(4.0, 9.0)))
        
        # Parks proximity (higher is better, 1-10 scale)
        factors['parks_proximity'] = min(10, max(1, random.uniform(3.0, 8.0)))
        
        # Noise level (lower is better, 1-10 scale)
        factors['noise_level'] = min(10, max(1, random.uniform(2.0, 7.0)))
        
        # Walkability (higher is better, 1-10 scale)
        factors['walkability'] = min(10, max(1, random.uniform(3.0, 9.0)))
        
        # Reset random seed
        random.seed()
        
        # Calculate weighted score (normalize to 0-100 scale)
        score = 0
        for factor, value in factors.items():
            if factor in weights:
                # For negative factors (where lower is better), invert the scale
                if weights[factor] < 0:
                    value = 11 - value  # Invert the scale (1 becomes 10, 10 becomes 1)
                    score += abs(weights[factor]) * value
                else:
                    score += weights[factor] * value
                    
        # Normalize to 0-100 scale
        score = round(score * 10, 1)
        
        return {
            'score': score,
            'message': "Location score calculated successfully",
            'factors': factors
        }
        
    except Exception as e:
        logger.error(f"Error calculating location score: {e}")
        return {
            'score': None,
            'message': f"Error calculating location score: {str(e)}",
            'factors': {}
        }

def get_nearby_amenities(latitude, longitude, radius_miles=1.0):
    """
    Find amenities near the property location.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        radius_miles: Search radius in miles
        
    Returns:
        dict: Nearby amenities by category
    """
    logger.info(f"Finding amenities near coordinates: {latitude}, {longitude} within {radius_miles} miles")
    
    if not latitude or not longitude:
        logger.warning("Missing coordinates for amenities search")
        return {
            'amenities': {},
            'message': "Amenities search requires valid coordinates"
        }
    
    # In a real implementation, we would query a GIS database here
    # For this demo, we'll generate randomized but realistic values
    try:
        # Use coordinates as seed for consistent results
        seed_value = abs(hash(f"{latitude}{longitude}{radius_miles}")) % 1000000
        random.seed(seed_value)
        
        amenities = {}
        
        # Generate random number of schools
        num_schools = random.randint(0, 4)
        amenities['schools'] = []
        school_types = ['Elementary', 'Middle', 'High', 'Private', 'Charter']
        school_names = ['Washington', 'Lincoln', 'Jefferson', 'Roosevelt', 'Kennedy', 'Madison', 'Franklin']
        
        for i in range(num_schools):
            school_type = random.choice(school_types)
            school_name = random.choice(school_names)
            distance = round(random.uniform(0.1, radius_miles), 2)
            rating = round(random.uniform(5.0, 9.5), 1)
            
            amenities['schools'].append({
                'name': f"{school_name} {school_type} School",
                'distance_miles': distance,
                'rating': rating
            })
        
        # Generate random number of parks
        num_parks = random.randint(0, 3)
        amenities['parks'] = []
        park_types = ['Neighborhood', 'Community', 'Regional', 'Nature']
        park_names = ['Green', 'Central', 'Riverside', 'Lakeside', 'Memorial', 'Heritage', 'Sunset']
        
        for i in range(num_parks):
            park_type = random.choice(park_types)
            park_name = random.choice(park_names)
            distance = round(random.uniform(0.1, radius_miles), 2)
            size_acres = round(random.uniform(1.5, 50.0), 1)
            
            amenities['parks'].append({
                'name': f"{park_name} {park_type} Park",
                'distance_miles': distance,
                'size_acres': size_acres
            })
        
        # Generate random number of shopping locations
        num_shopping = random.randint(0, 5)
        amenities['shopping'] = []
        shopping_types = ['Grocery', 'Mall', 'Convenience Store', 'Shopping Center', 'Specialty Shop']
        shopping_names = ['Marketplace', 'Village', 'Center', 'Corner', 'Plaza', 'Square']
        
        for i in range(num_shopping):
            shopping_type = random.choice(shopping_types)
            shopping_name = random.choice(shopping_names)
            distance = round(random.uniform(0.1, radius_miles), 2)
            
            amenities['shopping'].append({
                'name': f"{shopping_name} {shopping_type}",
                'distance_miles': distance
            })
        
        # Generate random number of restaurants
        num_restaurants = random.randint(0, 6)
        amenities['restaurants'] = []
        restaurant_types = ['Italian', 'Mexican', 'American', 'Chinese', 'Thai', 'Fast Food', 'Cafe']
        restaurant_names = ['Bella', 'El', 'The', 'Golden', 'Royal', 'Blue', 'Green']
        
        for i in range(num_restaurants):
            restaurant_type = random.choice(restaurant_types)
            restaurant_name = random.choice(restaurant_names)
            distance = round(random.uniform(0.1, radius_miles), 2)
            rating = round(random.uniform(3.0, 4.9), 1)
            
            amenities['restaurants'].append({
                'name': f"{restaurant_name} {restaurant_type}",
                'distance_miles': distance,
                'rating': rating
            })
        
        # Generate random number of transit stops
        num_transit = random.randint(0, 3)
        amenities['transit'] = []
        transit_types = ['Bus Stop', 'Subway Station', 'Light Rail', 'Commuter Rail']
        transit_lines = ['Red Line', 'Blue Line', 'Route 7', 'Express', 'Local']
        
        for i in range(num_transit):
            transit_type = random.choice(transit_types)
            transit_line = random.choice(transit_lines)
            distance = round(random.uniform(0.1, radius_miles), 2)
            
            amenities['transit'].append({
                'name': f"{transit_type} - {transit_line}",
                'distance_miles': distance
            })
        
        # Reset random seed
        random.seed()
        
        return {
            'amenities': amenities,
            'message': "Nearby amenities found successfully"
        }
        
    except Exception as e:
        logger.error(f"Error finding nearby amenities: {e}")
        return {
            'amenities': {},
            'message': f"Error finding nearby amenities: {str(e)}"
        }

def get_flood_risk_assessment(latitude, longitude):
    """
    Assess the flood risk for a property.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        
    Returns:
        dict: Flood risk assessment
    """
    logger.info(f"Assessing flood risk for coordinates: {latitude}, {longitude}")
    
    if not latitude or not longitude:
        logger.warning("Missing coordinates for flood risk assessment")
        return {
            'risk_level': None,
            'message': "Flood risk assessment requires valid coordinates"
        }
    
    # In a real implementation, we would query FEMA flood maps or similar
    # For this demo, we'll generate a consistent but randomized risk level
    try:
        # Use coordinates as seed for consistent results
        seed_value = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(seed_value)
        
        # Generate a random number between 0 and 100
        risk_factor = random.uniform(0, 100)
        
        # Define risk levels based on this factor
        if risk_factor < 70:
            risk_level = "Low"
            risk_description = "This property has minimal flood risk. It is not in a designated flood zone."
            insurance_recommendation = "Standard homeowner's insurance should be sufficient."
        elif risk_factor < 90:
            risk_level = "Moderate"
            risk_description = "This property has some flood risk. It may be near a designated flood zone."
            insurance_recommendation = "Consider supplemental flood insurance for added protection."
        else:
            risk_level = "High"
            risk_description = "This property has significant flood risk. It appears to be in a designated flood zone."
            insurance_recommendation = "Flood insurance is strongly recommended and may be required by lenders."
        
        # Reset random seed
        random.seed()
        
        return {
            'risk_level': risk_level,
            'risk_factor': round(risk_factor, 1),
            'description': risk_description,
            'insurance_recommendation': insurance_recommendation,
            'message': "Flood risk assessment completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error assessing flood risk: {e}")
        return {
            'risk_level': None,
            'message': f"Error assessing flood risk: {str(e)}"
        }

def get_school_district_info(latitude, longitude):
    """
    Get information about the school district serving a property.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        
    Returns:
        dict: School district information
    """
    logger.info(f"Getting school district info for coordinates: {latitude}, {longitude}")
    
    if not latitude or not longitude:
        logger.warning("Missing coordinates for school district lookup")
        return {
            'district_name': None,
            'message': "School district lookup requires valid coordinates"
        }
    
    # In a real implementation, we would query school district boundary data
    # For this demo, we'll generate randomized but realistic values
    try:
        # Use coordinates as seed for consistent results
        seed_value = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(seed_value)
        
        # Generate random district info
        district_names = ['Washington', 'Lincoln', 'Jefferson', 'Franklin', 'Roosevelt']
        district_types = ['Unified School District', 'Public Schools', 'School District']
        district_name = f"{random.choice(district_names)} {random.choice(district_types)}"
        
        # School ratings (1-10 scale)
        elementary_rating = round(random.uniform(5.0, 9.5), 1)
        middle_rating = round(random.uniform(5.0, 9.5), 1)
        high_rating = round(random.uniform(5.0, 9.5), 1)
        
        # District stats
        student_count = random.randint(5000, 25000)
        student_teacher_ratio = round(random.uniform(13.0, 22.0), 1)
        college_bound_rate = round(random.uniform(65.0, 95.0), 1)
        
        schools = {
            'elementary': {
                'name': f"{random.choice(district_names)} Elementary School",
                'distance_miles': round(random.uniform(0.2, 2.0), 1),
                'rating': elementary_rating,
                'grades': 'K-5'
            },
            'middle': {
                'name': f"{random.choice(district_names)} Middle School",
                'distance_miles': round(random.uniform(0.5, 3.0), 1),
                'rating': middle_rating,
                'grades': '6-8'
            },
            'high': {
                'name': f"{random.choice(district_names)} High School",
                'distance_miles': round(random.uniform(0.7, 4.0), 1),
                'rating': high_rating,
                'grades': '9-12'
            }
        }
        
        # Reset random seed
        random.seed()
        
        return {
            'district_name': district_name,
            'overall_rating': round((elementary_rating + middle_rating + high_rating) / 3, 1),
            'student_count': student_count,
            'student_teacher_ratio': student_teacher_ratio,
            'college_bound_rate': college_bound_rate,
            'schools': schools,
            'message': "School district information retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting school district info: {e}")
        return {
            'district_name': None,
            'message': f"Error getting school district info: {str(e)}"
        }

def get_crime_statistics(latitude, longitude, radius_miles=1.0):
    """
    Get crime statistics for a property's area.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        radius_miles: Radius for crime statistics
        
    Returns:
        dict: Crime statistics for the area
    """
    logger.info(f"Getting crime statistics for coordinates: {latitude}, {longitude}")
    
    if not latitude or not longitude:
        logger.warning("Missing coordinates for crime statistics lookup")
        return {
            'crime_index': None,
            'message': "Crime statistics lookup requires valid coordinates"
        }
    
    # In a real implementation, we would query crime databases
    # For this demo, we'll generate randomized but realistic values
    try:
        # Use coordinates as seed for consistent results
        seed_value = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(seed_value)
        
        # Generate a random safety score (higher is safer, 1-100 scale)
        safety_score = random.randint(50, 95)
        
        # Determine safety level
        if safety_score >= 80:
            safety_level = "Very Safe"
            compared_to_national = "significantly lower"
        elif safety_score >= 70:
            safety_level = "Safe"
            compared_to_national = "lower"
        elif safety_score >= 50:
            safety_level = "Average"
            compared_to_national = "about average"
        elif safety_score >= 30:
            safety_level = "Below Average"
            compared_to_national = "higher"
        else:
            safety_level = "Concerning"
            compared_to_national = "significantly higher"
        
        # Generate crime statistics by type
        crime_stats = {
            'violent': {
                'rate': round(random.uniform(1.0, 10.0) * (100 - safety_score) / 50, 2),
                'trend': random.choice(['increasing', 'decreasing', 'stable'])
            },
            'property': {
                'rate': round(random.uniform(5.0, 40.0) * (100 - safety_score) / 50, 2),
                'trend': random.choice(['increasing', 'decreasing', 'stable'])
            },
            'vehicle': {
                'rate': round(random.uniform(2.0, 20.0) * (100 - safety_score) / 50, 2),
                'trend': random.choice(['increasing', 'decreasing', 'stable'])
            }
        }
        
        # Reset random seed
        random.seed()
        
        return {
            'safety_score': safety_score,
            'safety_level': safety_level,
            'compared_to_national': compared_to_national,
            'crime_stats': crime_stats,
            'message': "Crime statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting crime statistics: {e}")
        return {
            'safety_score': None,
            'message': f"Error getting crime statistics: {str(e)}"
        }

def get_zoning_info(latitude, longitude):
    """
    Get zoning information for a property.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        
    Returns:
        dict: Zoning information
    """
    logger.info(f"Getting zoning information for coordinates: {latitude}, {longitude}")
    
    if not latitude or not longitude:
        logger.warning("Missing coordinates for zoning lookup")
        return {
            'zone_type': None,
            'message': "Zoning lookup requires valid coordinates"
        }
    
    # In a real implementation, we would query zoning databases
    # For this demo, we'll generate randomized but realistic values
    try:
        # Use coordinates as seed for consistent results
        seed_value = abs(hash(f"{latitude}{longitude}")) % 1000000
        random.seed(seed_value)
        
        # Generate random zoning info
        zone_types = [
            'R-1 (Single-Family Residential)',
            'R-2 (Medium Density Residential)',
            'R-3 (High Density Residential)',
            'C-1 (Neighborhood Commercial)',
            'C-2 (General Commercial)',
            'M-1 (Light Industrial)',
            'PUD (Planned Unit Development)'
        ]
        
        zone_type = random.choice(zone_types)
        
        # Generate restrictions based on zone type
        if 'R-1' in zone_type:
            restrictions = {
                'usage': 'Single-family residential only',
                'max_height': '35 feet',
                'min_lot_size': '5,000 sq ft',
                'setbacks': {
                    'front': '20 feet',
                    'side': '5 feet',
                    'rear': '15 feet'
                }
            }
        elif 'R-2' in zone_type:
            restrictions = {
                'usage': 'Single-family or multi-family residential',
                'max_height': '45 feet',
                'min_lot_size': '3,500 sq ft per unit',
                'setbacks': {
                    'front': '15 feet',
                    'side': '5 feet',
                    'rear': '15 feet'
                }
            }
        elif 'R-3' in zone_type:
            restrictions = {
                'usage': 'Multi-family residential',
                'max_height': '60 feet',
                'min_lot_size': '2,000 sq ft per unit',
                'setbacks': {
                    'front': '10 feet',
                    'side': '5 feet',
                    'rear': '10 feet'
                }
            }
        elif 'C-1' in zone_type:
            restrictions = {
                'usage': 'Small-scale retail and office',
                'max_height': '40 feet',
                'parking': '1 space per 300 sq ft',
                'setbacks': {
                    'front': '10 feet',
                    'side': '0-5 feet',
                    'rear': '10 feet'
                }
            }
        elif 'C-2' in zone_type:
            restrictions = {
                'usage': 'General commercial and retail',
                'max_height': '60 feet',
                'parking': '1 space per 250 sq ft',
                'setbacks': {
                    'front': '5 feet',
                    'side': '0 feet',
                    'rear': '5 feet'
                }
            }
        elif 'M-1' in zone_type:
            restrictions = {
                'usage': 'Light manufacturing and industrial',
                'max_height': '50 feet',
                'min_lot_size': '10,000 sq ft',
                'setbacks': {
                    'front': '20 feet',
                    'side': '10 feet',
                    'rear': '10 feet'
                }
            }
        else:  # PUD
            restrictions = {
                'usage': 'Mixed use as approved in development plan',
                'max_height': 'As approved in development plan',
                'min_lot_size': 'As approved in development plan',
                'setbacks': 'As approved in development plan'
            }
        
        # Reset random seed
        random.seed()
        
        return {
            'zone_type': zone_type,
            'restrictions': restrictions,
            'overlay_districts': random.choice([None, 'Historic District', 'Flood Hazard Overlay', 'Transit Oriented Development']),
            'message': "Zoning information retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting zoning information: {e}")
        return {
            'zone_type': None,
            'message': f"Error getting zoning information: {str(e)}"
        }

def coordinate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two coordinates in miles.
    Uses the Haversine formula for spherical distance.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        float: Distance in miles
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Earth radius in miles
    earth_radius = 3959.0
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c
    
    return distance