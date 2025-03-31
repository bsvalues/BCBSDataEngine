"""
Enhanced GIS Features for BCBS Values application.

This module provides functions for enhancing property valuations with GIS data.
"""
import logging
import math
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    
    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def calculate_proximity_features(
    latitude: float,
    longitude: float,
    poi_data: Dict[str, List[Dict[str, Union[str, float]]]]
) -> Dict[str, float]:
    """
    Calculate proximity features for a property based on its location and points of interest.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        poi_data: Dictionary of points of interest by category
        
    Returns:
        Dictionary of proximity features
    """
    features = {}
    
    # Initialize with defaults
    features['near_water'] = 0.0
    features['near_park'] = 0.0
    features['near_school'] = 0.0
    features['near_shopping'] = 0.0
    features['near_hospital'] = 0.0
    features['near_highway'] = 0.0
    features['near_transit'] = 0.0
    
    try:
        # Calculate distance to water bodies
        if 'water' in poi_data:
            water_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['water']
            ]
            min_water_distance = min(water_distances) if water_distances else 10.0
            features['near_water'] = calculate_proximity_score(min_water_distance, 2.0)
        
        # Calculate distance to parks
        if 'parks' in poi_data:
            park_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['parks']
            ]
            min_park_distance = min(park_distances) if park_distances else 10.0
            features['near_park'] = calculate_proximity_score(min_park_distance, 1.0)
        
        # Calculate distance to schools
        if 'schools' in poi_data:
            school_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['schools']
            ]
            min_school_distance = min(school_distances) if school_distances else 10.0
            features['near_school'] = calculate_proximity_score(min_school_distance, 1.5)
            
            # Also consider school quality if available
            if school_distances and min(school_distances) < 3.0:
                nearest_school_idx = school_distances.index(min(school_distances))
                nearest_school = poi_data['schools'][nearest_school_idx]
                school_quality = nearest_school.get('quality', 0.0)
                features['school_quality'] = float(school_quality) / 10.0  # Normalize to 0-1
        
        # Calculate distance to shopping centers
        if 'shopping' in poi_data:
            shopping_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['shopping']
            ]
            min_shopping_distance = min(shopping_distances) if shopping_distances else 10.0
            features['near_shopping'] = calculate_proximity_score(min_shopping_distance, 2.0)
        
        # Calculate distance to hospitals
        if 'hospitals' in poi_data:
            hospital_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['hospitals']
            ]
            min_hospital_distance = min(hospital_distances) if hospital_distances else 10.0
            features['near_hospital'] = calculate_proximity_score(min_hospital_distance, 3.0)
        
        # Calculate distance to highways
        if 'highways' in poi_data:
            highway_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['highways']
            ]
            min_highway_distance = min(highway_distances) if highway_distances else 10.0
            features['near_highway'] = calculate_proximity_score(min_highway_distance, 1.0)
            
            # Too close to highway is negative
            if min_highway_distance < 0.5:
                features['near_highway'] = -0.05
        
        # Calculate distance to transit stations
        if 'transit' in poi_data:
            transit_distances = [
                haversine_distance(latitude, longitude, poi['latitude'], poi['longitude'])
                for poi in poi_data['transit']
            ]
            min_transit_distance = min(transit_distances) if transit_distances else 10.0
            features['near_transit'] = calculate_proximity_score(min_transit_distance, 1.0)
    
    except Exception as e:
        logger.error(f"Error calculating proximity features: {str(e)}")
    
    return features


def calculate_proximity_score(distance: float, threshold: float) -> float:
    """
    Calculate a proximity score based on distance and threshold.
    
    Args:
        distance: Distance in kilometers
        threshold: Distance threshold for scoring
        
    Returns:
        Proximity score between 0 and 1
    """
    if distance > threshold * 2:
        return 0.0
    elif distance > threshold:
        return (threshold * 2 - distance) / threshold / 2
    else:
        return (threshold - distance) / threshold


def calculate_neighborhood_quality(
    latitude: float,
    longitude: float,
    neighborhood_data: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, float]:
    """
    Calculate neighborhood quality features.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        neighborhood_data: Dictionary of neighborhood data
        
    Returns:
        Dictionary of neighborhood quality features
    """
    features = {}
    
    # Default values
    features['neighborhood_quality'] = 0.5  # Neutral
    features['crime_rate'] = 0.5  # Neutral
    features['school_district_rating'] = 0.5  # Neutral
    features['walkability'] = 0.5  # Neutral
    
    try:
        if neighborhood_data:
            # Find the neighborhood that contains this point
            for neighborhood, data in neighborhood_data.items():
                if 'boundaries' in data:
                    if point_in_polygon(latitude, longitude, data['boundaries']):
                        features['neighborhood_quality'] = data.get('quality', 0.5)
                        features['crime_rate'] = data.get('crime_rate', 0.5)
                        features['school_district_rating'] = data.get('school_rating', 0.5)
                        features['walkability'] = data.get('walkability', 0.5)
                        break
    except Exception as e:
        logger.error(f"Error calculating neighborhood quality: {str(e)}")
    
    return features


def point_in_polygon(
    latitude: float,
    longitude: float,
    polygon: List[Tuple[float, float]]
) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        latitude: Point latitude
        longitude: Point longitude
        polygon: List of (lat, lon) tuples defining the polygon
        
    Returns:
        True if the point is inside the polygon, False otherwise
    """
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if longitude > min(p1x, p2x):
            if longitude <= max(p1x, p2x):
                if latitude <= max(p1y, p2y):
                    if p1x != p2x:
                        xinters = (longitude - p1x) * (p2y - p1y) / (p2x - p1x) + p1y
                    if p1y == p2y or latitude <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def load_gis_features(file_path: str) -> Dict:
    """
    Load GIS features from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary of GIS features
    """
    try:
        with open(file_path, 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        logger.error(f"Error loading GIS features from {file_path}: {str(e)}")
        return {}


def calculate_gis_adjustments(
    latitude: float,
    longitude: float,
    base_value: float,
    gis_features: Dict
) -> Tuple[float, Dict]:
    """
    Calculate value adjustments based on GIS features.
    
    Args:
        latitude: Property latitude
        longitude: Property longitude
        base_value: Base property value from the regression model
        gis_features: GIS feature data
        
    Returns:
        Tuple of (adjusted value, adjustment details)
    """
    adjustments = {}
    
    try:
        # Calculate proximity features
        poi_data = gis_features.get('points_of_interest', {})
        proximity_features = calculate_proximity_features(latitude, longitude, poi_data)
        
        # Calculate neighborhood quality
        neighborhood_data = gis_features.get('neighborhoods', {})
        neighborhood_features = calculate_neighborhood_quality(
            latitude, longitude, neighborhood_data
        )
        
        # Calculate overall quality score
        proximity_score = sum(proximity_features.values()) / max(1, len(proximity_features))
        quality_score = neighborhood_features.get('neighborhood_quality', 0.5)
        
        # Calculate adjustments
        proximity_adjustment = (proximity_score - 0.5) * 0.10  # -5% to +5%
        quality_adjustment = (quality_score - 0.5) * 0.20  # -10% to +10%
        
        # Store adjustments
        adjustments = {
            'proximity_adjustment': proximity_adjustment,
            'quality_adjustment': quality_adjustment,
            'proximity_features': proximity_features,
            'neighborhood_features': neighborhood_features,
            'base_value': base_value
        }
        
        # Apply adjustments
        adjusted_value = base_value * (1 + proximity_adjustment + quality_adjustment)
    
    except Exception as e:
        logger.error(f"Error calculating GIS adjustments: {str(e)}")
        adjusted_value = base_value
    
    return adjusted_value, adjustments