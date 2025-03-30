"""
Enhanced GIS data integration for BCBS Values.

This module provides advanced GIS (Geographic Information System) functionality
for the BCBS Values real estate valuation system, enabling more sophisticated
spatial analysis and location-based pricing adjustments.

Key enhancements include:
1. Improved proximity scoring with exponential decay functions
2. Weighted multi-factor location scoring
3. School district and quality integration
4. Flood zone risk assessment
5. Walkability and amenity scoring
6. Traffic and noise impact assessment
7. View quality estimation
8. Future development potential
"""

import math
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Constants for GIS calculations
EARTH_RADIUS_KM = 6371  # Earth radius in kilometers
KM_PER_DEGREE_LAT = 111.32  # km per degree of latitude (approximate)
FLOOD_ZONE_RISK_WEIGHTS = {
    'X': 1.0,       # Minimal flood risk
    'X500': 0.95,   # 0.2% annual chance flood
    'AE': 0.85,     # 1% annual chance flood
    'A': 0.80,      # 1% annual chance flood, no base flood elevation
    'VE': 0.75      # High coastal flood risk
}

# School rating impact on home values (percent value increase per rating point)
SCHOOL_RATING_IMPACT = 0.015  # 1.5% per point

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (in decimal degrees)
        lat2, lon2: Latitude and longitude of point 2 (in decimal degrees)
        
    Returns:
        float: Distance between the points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Distance in kilometers
    return c * EARTH_RADIUS_KM

def calculate_proximity_score(lat, lon, ref_points, use_exponential_decay=True):
    """
    Calculate a sophisticated proximity score based on distance to reference points.
    
    This function uses an exponential decay function to model the diminishing
    impact of distance on property value, which better reflects real-world
    property valuation than linear adjustments.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        ref_points (dict): Dictionary of reference points with coordinates and weights
        use_exponential_decay (bool): Whether to use exponential decay for distance impact
        
    Returns:
        dict: Dictionary with proximity scores for each point and a combined score
    """
    if not ref_points:
        return {'combined_score': 0}
    
    results = {}
    weighted_scores = []
    total_weight = 0
    
    for point_name, point_info in ref_points.items():
        if 'lat' not in point_info or 'lon' not in point_info:
            continue
            
        # Calculate distance
        dist_km = haversine_distance(lat, lon, point_info['lat'], point_info['lon'])
        
        # Get point weight (importance)
        weight = point_info.get('weight', 1.0)
        total_weight += weight
        
        # Calculate score based on distance
        if use_exponential_decay:
            # Exponential decay function: score = e^(-distance/scale_factor)
            # This creates a more realistic proximity impact that diminishes with distance
            scale_factor = point_info.get('scale_factor', 2.0)  # km
            score = math.exp(-dist_km / scale_factor)
        else:
            # Linear decay with cutoff (simpler model)
            max_distance = point_info.get('max_distance', 10.0)  # km
            score = max(0, 1 - (dist_km / max_distance))
        
        # Apply weight and store
        weighted_score = score * weight
        weighted_scores.append(weighted_score)
        
        # Store individual result
        results[f"proximity_to_{point_name}"] = {
            'distance_km': dist_km,
            'score': score,
            'weighted_score': weighted_score
        }
    
    # Calculate combined score (weighted average)
    if total_weight > 0:
        results['combined_score'] = sum(weighted_scores) / total_weight
    else:
        results['combined_score'] = 0
        
    return results

def assess_flood_zone_risk(lat, lon, flood_zone_data=None):
    """
    Assess flood zone risk and its impact on property value.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        flood_zone_data (pd.DataFrame, optional): DataFrame with flood zone polygons or points
        
    Returns:
        dict: Flood zone assessment with risk score and value impact
    """
    # Default to minimal risk if no flood data available
    if flood_zone_data is None:
        return {
            'zone': 'Unknown',
            'risk_level': 'Unknown',
            'value_multiplier': 1.0
        }
    
    try:
        # Find the closest flood zone point/polygon to the property
        # This is a simplified implementation - a real-world solution would use
        # polygon containment checks with shapely/geopandas
        
        # Create a dataframe for the property point
        property_point = pd.DataFrame({
            'latitude': [lat],
            'longitude': [lon]
        })
        
        # Find closest flood zone record
        # For this simplified version, assume flood_zone_data has lat/lon columns
        # A more robust implementation would use proper GIS libraries
        closest_idx = ((flood_zone_data['latitude'] - lat)**2 + 
                       (flood_zone_data['longitude'] - lon)**2).idxmin()
        
        zone = flood_zone_data.loc[closest_idx, 'flood_zone']
        
        # Map zone to risk value
        value_multiplier = FLOOD_ZONE_RISK_WEIGHTS.get(zone, 1.0)
        
        # Determine risk level based on multiplier
        if value_multiplier >= 0.95:
            risk_level = 'Minimal'
        elif value_multiplier >= 0.85:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
            
        return {
            'zone': zone,
            'risk_level': risk_level,
            'value_multiplier': value_multiplier
        }
        
    except Exception as e:
        logger.warning(f"Error assessing flood zone: {str(e)}")
        return {
            'zone': 'Error',
            'risk_level': 'Unknown',
            'value_multiplier': 1.0
        }

def calculate_school_quality_impact(lat, lon, school_data=None):
    """
    Calculate impact of school quality on property value.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        school_data (pd.DataFrame, optional): DataFrame with school locations and ratings
        
    Returns:
        dict: School quality assessment with value impact
    """
    # Default if no school data available
    if school_data is None:
        return {
            'elementary': None,
            'middle': None,
            'high': None,
            'avg_rating': None,
            'value_multiplier': 1.0
        }
    
    try:
        # Dictionary to store results by school type
        schools = {
            'elementary': None,
            'middle': None,
            'high': None
        }
        
        # Find closest school of each type within a reasonable radius (e.g., 10km)
        max_distance = 10  # km
        
        for school_type in schools.keys():
            # Filter to this school type
            type_schools = school_data[school_data['type'] == school_type]
            
            if len(type_schools) == 0:
                continue
                
            # Calculate distances to all schools of this type
            distances = []
            for idx, school in type_schools.iterrows():
                dist = haversine_distance(lat, lon, school['latitude'], school['longitude'])
                distances.append((idx, dist))
            
            # Find closest school within limit
            closest = min(distances, key=lambda x: x[1])
            idx, dist = closest
            
            if dist <= max_distance:
                school = type_schools.loc[idx]
                schools[school_type] = {
                    'name': school['name'],
                    'rating': school['rating'],
                    'distance_km': dist
                }
        
        # Calculate average rating of assigned schools
        ratings = [schools[s]['rating'] for s in schools if schools[s] is not None]
        
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            
            # Calculate value impact (baseline = 5 on scale of 1-10)
            # Each point above 5 adds SCHOOL_RATING_IMPACT percent to value
            # Each point below 5 subtracts SCHOOL_RATING_IMPACT percent from value
            baseline = 5.0
            value_multiplier = 1.0 + (avg_rating - baseline) * SCHOOL_RATING_IMPACT
            
            return {
                'elementary': schools['elementary'],
                'middle': schools['middle'],
                'high': schools['high'],
                'avg_rating': avg_rating,
                'value_multiplier': value_multiplier
            }
        else:
            return {
                'elementary': None,
                'middle': None,
                'high': None,
                'avg_rating': None,
                'value_multiplier': 1.0
            }
            
    except Exception as e:
        logger.warning(f"Error calculating school impact: {str(e)}")
        return {
            'elementary': None,
            'middle': None,
            'high': None,
            'avg_rating': None,
            'value_multiplier': 1.0
        }

def estimate_view_quality(lat, lon, elevation_data=None, dem_data=None, landmarks=None):
    """
    Estimate view quality based on elevation, digital elevation model, and landmarks.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        elevation_data (pd.DataFrame, optional): DataFrame with elevation points
        dem_data (numpy.ndarray, optional): Digital Elevation Model raster data
        landmarks (dict, optional): Dictionary of landmarks with coordinates and values
        
    Returns:
        dict: View quality assessment with value impact
    """
    # Default to neutral impact if no data available
    if elevation_data is None and dem_data is None:
        return {
            'elevation': None,
            'relative_height': None,
            'landmark_views': [],
            'view_score': 0.5,
            'value_multiplier': 1.0
        }
    
    try:
        # Simplified implementation - just check if property is higher than surrounding area
        # A real implementation would use viewshed analysis from GIS tools
        
        elevation = None
        relative_height = 0
        landmark_views = []
        
        # 1. Get property elevation if elevation data is available
        if elevation_data is not None:
            # Find closest elevation point
            closest_idx = ((elevation_data['latitude'] - lat)**2 + 
                          (elevation_data['longitude'] - lon)**2).idxmin()
            elevation = elevation_data.loc[closest_idx, 'elevation']
            
            # Calculate average elevation in surrounding area (1km radius)
            nearby = elevation_data[
                (elevation_data['latitude'] - lat)**2 + 
                (elevation_data['longitude'] - lon)**2 <= (1/111)**2  # ~1km radius
            ]
            
            if len(nearby) > 1:  # More than just this point
                avg_elevation = nearby['elevation'].mean()
                relative_height = elevation - avg_elevation
        
        # 2. Check for landmark views if landmark data is available
        # (In reality, would use terrain viewshed analysis)
        if landmarks is not None and elevation is not None:
            for name, landmark in landmarks.items():
                if 'lat' in landmark and 'lon' in landmark and 'elevation' in landmark:
                    # Calculate distance and direction to landmark
                    dist = haversine_distance(lat, lon, landmark['lat'], landmark['lon'])
                    
                    # Simplified line-of-sight check 
                    # (actual implementation would use elevation profile)
                    if dist < landmark.get('max_view_distance', 30) and elevation > landmark['elevation']:
                        view_quality = max(0, 1 - (dist / landmark.get('max_view_distance', 30)))
                        landmark_views.append({
                            'landmark': name,
                            'distance_km': dist,
                            'view_quality': view_quality
                        })
        
        # 3. Calculate overall view score and value impact
        # Base score on relative height and landmark views
        view_score = 0.5  # Neutral starting point
        
        # Adjust for relative height (being higher than surroundings is good)
        if relative_height is not None:
            # Normalize to a reasonable range (-50m to +50m)
            height_factor = min(0.25, max(-0.25, relative_height / 200))
            view_score += height_factor
        
        # Adjust for landmark views
        if landmark_views:
            # Take the best landmark view
            best_view = max(landmark_views, key=lambda x: x['view_quality'])
            view_score += best_view['view_quality'] * 0.25
        
        # Clamp to 0-1 range
        view_score = min(1.0, max(0.0, view_score))
        
        # Calculate value impact (0.9 to 1.2 multiplier)
        value_multiplier = 0.9 + (0.3 * view_score)
        
        return {
            'elevation': elevation,
            'relative_height': relative_height,
            'landmark_views': landmark_views,
            'view_score': view_score,
            'value_multiplier': value_multiplier
        }
        
    except Exception as e:
        logger.warning(f"Error estimating view quality: {str(e)}")
        return {
            'elevation': None,
            'relative_height': None,
            'landmark_views': [],
            'view_score': 0.5,
            'value_multiplier': 1.0
        }

def calculate_amenity_score(lat, lon, amenities_data=None):
    """
    Calculate amenity score based on nearby points of interest.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        amenities_data (pd.DataFrame, optional): DataFrame with amenity points
        
    Returns:
        dict: Amenity assessment with value impact
    """
    # Default if no amenity data available
    if amenities_data is None:
        return {
            'nearby_amenities': [],
            'amenity_score': 0.5,
            'value_multiplier': 1.0
        }
    
    try:
        # Define amenity weights and max distances
        amenity_weights = {
            'grocery': {'weight': 0.8, 'max_dist': 2.0},
            'restaurant': {'weight': 0.6, 'max_dist': 3.0},
            'school': {'weight': 0.7, 'max_dist': 2.0},
            'park': {'weight': 0.7, 'max_dist': 1.5},
            'hospital': {'weight': 0.5, 'max_dist': 5.0},
            'shopping': {'weight': 0.6, 'max_dist': 3.0},
            'transit': {'weight': 0.7, 'max_dist': 1.0},
            'gym': {'weight': 0.4, 'max_dist': 2.0},
            'pharmacy': {'weight': 0.5, 'max_dist': 2.0},
            'library': {'weight': 0.4, 'max_dist': 3.0}
        }
        
        # Find nearby amenities for each type
        nearby_amenities = []
        amenity_scores = []
        total_weight = 0
        
        for amenity_type, settings in amenity_weights.items():
            # Filter to this amenity type
            type_amenities = amenities_data[amenities_data['type'] == amenity_type]
            
            if len(type_amenities) == 0:
                continue
                
            # Find closest amenity of this type
            closest_dist = float('inf')
            closest_name = None
            
            for idx, amenity in type_amenities.iterrows():
                dist = haversine_distance(lat, lon, amenity['latitude'], amenity['longitude'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_name = amenity['name']
            
            # Calculate score if within max distance
            if closest_dist <= settings['max_dist']:
                # Normalize distance (1.0 at 0km, 0.0 at max_dist)
                score = max(0, 1 - (closest_dist / settings['max_dist']))
                
                # Store nearby amenity
                nearby_amenities.append({
                    'type': amenity_type,
                    'name': closest_name,
                    'distance_km': closest_dist,
                    'score': score
                })
                
                # Add to weighted scores
                weight = settings['weight']
                amenity_scores.append(score * weight)
                total_weight += weight
        
        # Calculate overall amenity score
        if total_weight > 0:
            amenity_score = sum(amenity_scores) / total_weight
        else:
            amenity_score = 0.5  # Neutral score if no data
            
        # Calculate value impact (0.9 to 1.1 multiplier)
        value_multiplier = 0.9 + (0.2 * amenity_score)
        
        return {
            'nearby_amenities': nearby_amenities,
            'amenity_score': amenity_score,
            'value_multiplier': value_multiplier
        }
        
    except Exception as e:
        logger.warning(f"Error calculating amenity score: {str(e)}")
        return {
            'nearby_amenities': [],
            'amenity_score': 0.5,
            'value_multiplier': 1.0
        }

def estimate_traffic_noise_impact(lat, lon, road_data=None, traffic_data=None):
    """
    Estimate traffic and noise impact on property value.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        road_data (pd.DataFrame, optional): DataFrame with road segments and classifications
        traffic_data (pd.DataFrame, optional): DataFrame with traffic volume data
        
    Returns:
        dict: Traffic and noise assessment with value impact
    """
    # Default if no road data available
    if road_data is None:
        return {
            'nearest_road': None,
            'noise_level': 'Unknown',
            'traffic_score': 0.5,
            'value_multiplier': 1.0
        }
    
    try:
        # Define road type weights for noise/traffic impact
        road_impacts = {
            'highway': {'noise_factor': 0.8, 'buffer_distance': 0.5},  # 500m
            'arterial': {'noise_factor': 0.5, 'buffer_distance': 0.2},  # 200m
            'collector': {'noise_factor': 0.3, 'buffer_distance': 0.1},  # 100m
            'local': {'noise_factor': 0.1, 'buffer_distance': 0.05}   # 50m
        }
        
        # Find nearest road of each type
        nearest_roads = []
        
        for road_type, impact in road_impacts.items():
            # Filter to this road type
            type_roads = road_data[road_data['type'] == road_type]
            
            if len(type_roads) == 0:
                continue
                
            # Find distance to nearest road of this type
            # This is a simplified implementation - a real solution would
            # calculate distance to road segments using shapely
            nearest_dist = float('inf')
            nearest_id = None
            
            for idx, road in type_roads.iterrows():
                # Simple distance to point (an actual implementation would use line distance)
                dist = haversine_distance(lat, lon, road['latitude'], road['longitude'])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = road['id']
            
            if nearest_dist < impact['buffer_distance'] * 2:  # Within 2x buffer
                # Calculate impact based on distance
                proximity = max(0, 1 - (nearest_dist / impact['buffer_distance']))
                
                # Apply noise factor
                impact_score = proximity * impact['noise_factor']
                
                nearest_roads.append({
                    'type': road_type,
                    'id': nearest_id,
                    'distance_km': nearest_dist,
                    'impact_score': impact_score
                })
        
        # Determine overall traffic impact
        if nearest_roads:
            # Take highest impact road
            highest_impact = max(nearest_roads, key=lambda x: x['impact_score'])
            traffic_score = 1 - highest_impact['impact_score']  # Invert (lower is worse)
            
            # Determine noise level category
            if traffic_score < 0.3:
                noise_level = 'High'
            elif traffic_score < 0.7:
                noise_level = 'Moderate'
            else:
                noise_level = 'Low'
                
            # Calculate value impact (0.85 to 1.05 multiplier)
            # Traffic has less impact than other factors but still matters
            value_multiplier = 0.85 + (0.2 * traffic_score)
            
            return {
                'nearest_road': highest_impact,
                'other_roads': [r for r in nearest_roads if r != highest_impact],
                'noise_level': noise_level,
                'traffic_score': traffic_score,
                'value_multiplier': value_multiplier
            }
        else:
            # No nearby roads - great for value!
            return {
                'nearest_road': None,
                'noise_level': 'Minimal',
                'traffic_score': 0.95,
                'value_multiplier': 1.05
            }
            
    except Exception as e:
        logger.warning(f"Error estimating traffic impact: {str(e)}")
        return {
            'nearest_road': None,
            'noise_level': 'Unknown',
            'traffic_score': 0.5,
            'value_multiplier': 1.0
        }

def analyze_housing_density(lat, lon, properties_df, radius_km=1.0):
    """
    Analyze housing density in the surrounding area and its impact on value.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        properties_df (pd.DataFrame): DataFrame with all properties and coordinates
        radius_km (float): Radius in kilometers to analyze
        
    Returns:
        dict: Housing density analysis with value impact
    """
    try:
        if 'latitude' not in properties_df.columns or 'longitude' not in properties_df.columns:
            return {
                'density': None,
                'relative_density': None,
                'value_multiplier': 1.0
            }
        
        # Calculate distances to all other properties
        distances = []
        for idx, prop in properties_df.iterrows():
            # Skip if missing coordinates
            if pd.isna(prop['latitude']) or pd.isna(prop['longitude']):
                continue
                
            dist = haversine_distance(lat, lon, prop['latitude'], prop['longitude'])
            distances.append(dist)
        
        # Count properties within radius
        nearby = sum(1 for d in distances if d <= radius_km)
        
        # Calculate density (properties per sq km)
        area = math.pi * radius_km**2  # Area in sq km
        density = nearby / area
        
        # Calculate density percentile relative to the whole dataset
        # First, calculate density for each property
        if len(properties_df) > 100:  # Only do this for reasonably sized datasets
            all_densities = []
            sample_size = min(len(properties_df), 200)  # Cap for performance
            sample = properties_df.sample(n=sample_size)
            
            for idx, prop in sample.iterrows():
                if pd.isna(prop['latitude']) or pd.isna(prop['longitude']):
                    continue
                    
                # Count nearby properties
                prop_nearby = sum(1 for idx2, prop2 in properties_df.iterrows()
                                if not pd.isna(prop2['latitude']) and not pd.isna(prop2['longitude']) and
                                haversine_distance(prop['latitude'], prop['longitude'], 
                                                  prop2['latitude'], prop2['longitude']) <= radius_km)
                prop_density = prop_nearby / area
                all_densities.append(prop_density)
            
            if all_densities:
                # Calculate percentile of this property's density
                percentile = sum(1 for d in all_densities if d <= density) / len(all_densities)
                
                # Determine relative density
                if percentile < 0.25:
                    relative_density = 'Low'
                elif percentile < 0.75:
                    relative_density = 'Medium'
                else:
                    relative_density = 'High'
                    
                # Calculate value impact
                # We assume a bell curve where medium density is best
                # Both very low and very high density reduce value
                optimal = 0.5  # 50th percentile is optimal
                deviation = abs(percentile - optimal)
                
                # Convert to multiplier (0.95 to 1.05 range)
                value_multiplier = 1.05 - (deviation * 0.2)
                
                return {
                    'density': density,
                    'percentile': percentile,
                    'relative_density': relative_density,
                    'nearby_properties': nearby,
                    'value_multiplier': value_multiplier
                }
        
        # Fallback if we couldn't calculate percentiles
        return {
            'density': density,
            'relative_density': 'Unknown',
            'nearby_properties': nearby,
            'value_multiplier': 1.0
        }
            
    except Exception as e:
        logger.warning(f"Error analyzing housing density: {str(e)}")
        return {
            'density': None,
            'relative_density': None,
            'value_multiplier': 1.0
        }

def evaluate_future_development(lat, lon, zoning_data=None, development_projects=None):
    """
    Evaluate future development potential and impact on property value.
    
    Args:
        lat (float): Latitude of the property
        lon (float): Longitude of the property
        zoning_data (pd.DataFrame, optional): DataFrame with zoning information
        development_projects (pd.DataFrame, optional): DataFrame with planned developments
        
    Returns:
        dict: Future development assessment with value impact
    """
    # Default if no zoning/development data available
    if zoning_data is None and development_projects is None:
        return {
            'zoning': None,
            'nearby_developments': [],
            'growth_potential': 'Unknown',
            'value_multiplier': 1.0
        }
    
    try:
        # Get property zoning if available
        zoning = None
        if zoning_data is not None:
            # Find zoning for this location
            # In a real implementation, this would use polygon containment checks
            closest_idx = ((zoning_data['latitude'] - lat)**2 + 
                          (zoning_data['longitude'] - lon)**2).idxmin()
            zoning = zoning_data.loc[closest_idx, 'zoning']
        
        # Check for nearby development projects
        nearby_developments = []
        if development_projects is not None:
            for idx, project in development_projects.iterrows():
                dist = haversine_distance(lat, lon, project['latitude'], project['longitude'])
                
                # Consider projects within 3km
                if dist <= 3:
                    impact_score = 0
                    
                    # Evaluate impact type (positive/negative)
                    if project['type'] in ['residential', 'mixed_use', 'commercial_retail', 'park']:
                        # These generally increase surrounding property values
                        impact_score = max(0, 0.5 - (dist / 6))  # 0.5 at 0km, 0 at 3km
                    elif project['type'] in ['industrial', 'utility', 'waste_management']:
                        # These generally decrease surrounding property values
                        impact_score = min(0, -0.5 + (dist / 6))  # -0.5 at 0km, 0 at 3km
                        
                    nearby_developments.append({
                        'name': project['name'],
                        'type': project['type'],
                        'distance_km': dist,
                        'impact_score': impact_score,
                        'completion_year': project.get('completion_year')
                    })
        
        # Calculate combined development impact
        development_impact = 0
        if nearby_developments:
            # Sum impacts, weighted by distance
            development_impact = sum(d['impact_score'] for d in nearby_developments)
            
        # Determine growth potential based on zoning and nearby developments
        growth_potential = 'Stable'  # Default
        
        if zoning in ['R1', 'R2', 'SF', 'SFR', 'rural']:
            # Low-density residential has limited growth
            base_growth = 0.02  # 2% annual appreciation
        elif zoning in ['R3', 'MFR', 'MDR', 'HDR', 'MF']:
            # Medium to high density residential has good growth
            base_growth = 0.04  # 4% annual appreciation
        elif zoning in ['MU', 'mixed', 'CBD', 'CC', 'downtown']:
            # Mixed-use and commercial centers have highest growth
            base_growth = 0.05  # 5% annual appreciation
        elif zoning in ['commercial', 'retail', 'office', 'CR']:
            # Commercial generally has good growth
            base_growth = 0.04  # 4% annual appreciation
        elif zoning in ['industrial', 'manufacturing', 'I1', 'I2']:
            # Industrial areas vary greatly
            base_growth = 0.03  # 3% annual appreciation
        else:
            # Unknown or agricultural
            base_growth = 0.02  # 2% annual appreciation
        
        # Adjust base growth by development impact
        adjusted_growth = base_growth + development_impact
        
        # Categorize growth potential
        if adjusted_growth < 0.02:
            growth_potential = 'Limited'
        elif adjusted_growth < 0.04:
            growth_potential = 'Moderate'
        elif adjusted_growth < 0.06:
            growth_potential = 'Strong'
        else:
            growth_potential = 'Excellent'
        
        # Calculate value multiplier
        # Higher growth potential areas command premium prices
        value_multiplier = 0.95 + (adjusted_growth * 5)  # 0.95 to 1.25 range
        
        return {
            'zoning': zoning,
            'nearby_developments': nearby_developments,
            'development_impact': development_impact,
            'base_appreciation': base_growth,
            'adjusted_appreciation': adjusted_growth,
            'growth_potential': growth_potential,
            'value_multiplier': value_multiplier
        }
        
    except Exception as e:
        logger.warning(f"Error evaluating future development: {str(e)}")
        return {
            'zoning': None,
            'nearby_developments': [],
            'growth_potential': 'Unknown',
            'value_multiplier': 1.0
        }

def calculate_combined_gis_multiplier(gis_factors):
    """
    Calculate a combined GIS price multiplier from multiple factors.
    
    Args:
        gis_factors (dict): Dictionary of GIS factors with value multipliers
        
    Returns:
        float: Combined value multiplier
    """
    # Extract value multipliers from factors
    multipliers = {
        # Essential factors - higher weights
        'proximity': gis_factors.get('proximity', {}).get('value_multiplier', 1.0),
        'schools': gis_factors.get('schools', {}).get('value_multiplier', 1.0),
        'view': gis_factors.get('view', {}).get('value_multiplier', 1.0),
        'amenities': gis_factors.get('amenities', {}).get('value_multiplier', 1.0),
        
        # Secondary factors - lower weights
        'flood': gis_factors.get('flood', {}).get('value_multiplier', 1.0),
        'traffic': gis_factors.get('traffic', {}).get('value_multiplier', 1.0),
        'density': gis_factors.get('density', {}).get('value_multiplier', 1.0),
        'development': gis_factors.get('development', {}).get('value_multiplier', 1.0)
    }
    
    # Factor weights - must sum to 1.0
    weights = {
        'proximity': 0.25,
        'schools': 0.20,
        'view': 0.15,
        'amenities': 0.15,
        'flood': 0.10,
        'traffic': 0.05,
        'density': 0.05,
        'development': 0.05
    }
    
    # Calculate weighted average multiplier
    combined_multiplier = 1.0
    weight_sum = 0
    
    for factor, multiplier in multipliers.items():
        if multiplier is not None:
            weight = weights[factor]
            combined_multiplier += (multiplier - 1.0) * weight
            weight_sum += weight
    
    # Normalize in case not all factors were available
    if weight_sum > 0 and weight_sum < 1.0:
        # Scale the adjustment to account for missing factors
        adjustment = (combined_multiplier - 1.0) * (1.0 / weight_sum)
        combined_multiplier = 1.0 + adjustment
    
    # Clamp to reasonable range (0.7 to 1.3)
    return min(1.3, max(0.7, combined_multiplier))

def enhance_property_with_gis(property_data, gis_datasets=None):
    """
    Enhance a property record with comprehensive GIS analysis.
    
    Args:
        property_data (dict): Property data with latitude and longitude
        gis_datasets (dict, optional): Dictionary of GIS datasets
            - ref_points: dict of reference points
            - neighborhood_ratings: dict of neighborhood ratings
            - flood_zones: DataFrame of flood zone data
            - schools: DataFrame of school data
            - amenities: DataFrame of amenity data
            - elevation: DataFrame of elevation data
            - landmarks: dict of landmark data
            - roads: DataFrame of road data
            - zoning: DataFrame of zoning data
            - developments: DataFrame of development project data
            - properties: DataFrame of all properties for density analysis
            
    Returns:
        dict: Enhanced property data with GIS features and value factors
    """
    # Default GIS datasets to empty dict
    if gis_datasets is None:
        gis_datasets = {}
    
    try:
        # Ensure we have coordinates
        lat = property_data.get('latitude')
        lon = property_data.get('longitude')
        
        if lat is None or lon is None:
            logger.warning("Property missing coordinates, cannot perform GIS analysis")
            return property_data
        
        # Initialize GIS factors dictionary
        gis_factors = {}
        
        # 1. Calculate proximity score
        ref_points = gis_datasets.get('ref_points')
        if ref_points:
            proximity = calculate_proximity_score(lat, lon, ref_points)
            gis_factors['proximity'] = {
                'scores': proximity,
                'value_multiplier': 0.9 + (proximity['combined_score'] * 0.2)  # 0.9 to 1.1
            }
        
        # 2. Apply neighborhood ratings
        neighborhood_ratings = gis_datasets.get('neighborhood_ratings')
        city = property_data.get('city')
        neighborhood = property_data.get('neighborhood')
        
        if neighborhood_ratings and (city or neighborhood):
            # Try to match by neighborhood, then by city
            rating = neighborhood_ratings.get(neighborhood, neighborhood_ratings.get(city))
            
            if rating:
                gis_factors['neighborhood'] = {
                    'rating': rating,
                    'value_multiplier': rating  # Directly use rating as multiplier
                }
        
        # 3. Assess flood zone risk
        flood_zones = gis_datasets.get('flood_zones')
        if flood_zones is not None:
            flood_assessment = assess_flood_zone_risk(lat, lon, flood_zones)
            gis_factors['flood'] = flood_assessment
        
        # 4. Calculate school quality impact
        schools = gis_datasets.get('schools')
        if schools is not None:
            school_assessment = calculate_school_quality_impact(lat, lon, schools)
            gis_factors['schools'] = school_assessment
        
        # 5. Estimate view quality
        elevation_data = gis_datasets.get('elevation')
        landmarks = gis_datasets.get('landmarks')
        view_assessment = estimate_view_quality(lat, lon, elevation_data, None, landmarks)
        gis_factors['view'] = view_assessment
        
        # 6. Calculate amenity score
        amenities = gis_datasets.get('amenities')
        if amenities is not None:
            amenity_assessment = calculate_amenity_score(lat, lon, amenities)
            gis_factors['amenities'] = amenity_assessment
        
        # 7. Estimate traffic and noise impact
        roads = gis_datasets.get('roads')
        if roads is not None:
            traffic_assessment = estimate_traffic_noise_impact(lat, lon, roads)
            gis_factors['traffic'] = traffic_assessment
        
        # 8. Analyze housing density
        properties_df = gis_datasets.get('properties')
        if properties_df is not None and len(properties_df) > 10:
            density_assessment = analyze_housing_density(lat, lon, properties_df)
            gis_factors['density'] = density_assessment
        
        # 9. Evaluate future development
        zoning = gis_datasets.get('zoning')
        developments = gis_datasets.get('developments')
        if zoning is not None or developments is not None:
            development_assessment = evaluate_future_development(lat, lon, zoning, developments)
            gis_factors['development'] = development_assessment
        
        # 10. Calculate combined GIS multiplier
        combined_multiplier = calculate_combined_gis_multiplier(gis_factors)
        
        # 11. Add GIS data to property
        enhanced_property = property_data.copy()
        enhanced_property['gis_analysis'] = gis_factors
        enhanced_property['gis_value_multiplier'] = combined_multiplier
        
        # 12. Add gis_location_quality for compatibility
        if 'proximity' in gis_factors:
            proximity_score = gis_factors['proximity']['scores']['combined_score']
            enhanced_property['gis_location_quality'] = proximity_score
            
        # 13. Generate summary for reporting
        gis_summary = {
            'proximity_score': gis_factors.get('proximity', {}).get('scores', {}).get('combined_score'),
            'school_rating': gis_factors.get('schools', {}).get('avg_rating'),
            'flood_risk': gis_factors.get('flood', {}).get('risk_level'),
            'view_score': gis_factors.get('view', {}).get('view_score'),
            'amenity_score': gis_factors.get('amenities', {}).get('amenity_score'),
            'noise_level': gis_factors.get('traffic', {}).get('noise_level'),
            'growth_potential': gis_factors.get('development', {}).get('growth_potential'),
            'combined_multiplier': combined_multiplier
        }
        enhanced_property['gis_summary'] = gis_summary
        
        logger.info(f"Enhanced property with GIS analysis, value multiplier: {combined_multiplier:.4f}")
        return enhanced_property
        
    except Exception as e:
        logger.error(f"Error in GIS enhancement: {str(e)}", exc_info=True)
        return property_data  # Return original data on error

def process_properties_with_gis(properties_df, gis_datasets=None):
    """
    Process multiple properties with GIS enhancement.
    
    Args:
        properties_df (pd.DataFrame): DataFrame of properties
        gis_datasets (dict, optional): Dictionary of GIS datasets
            
    Returns:
        pd.DataFrame: Enhanced properties DataFrame with GIS features
    """
    try:
        # Create copy to avoid modifying original
        df = properties_df.copy()
        
        # Add gis_datasets['properties'] for density calculations
        if gis_datasets is None:
            gis_datasets = {}
            
        gis_datasets['properties'] = properties_df
        
        # Track GIS multipliers for column addition
        gis_multipliers = []
        
        # Process each property
        logger.info(f"Processing {len(df)} properties with GIS enhancement")
        
        for idx, row in df.iterrows():
            # Skip if missing coordinates
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                gis_multipliers.append(1.0)
                continue
                
            # Convert row to dict for processing
            property_dict = row.to_dict()
            
            # Enhance with GIS
            enhanced = enhance_property_with_gis(property_dict, gis_datasets)
            
            # Store multiplier
            multiplier = enhanced.get('gis_value_multiplier', 1.0)
            gis_multipliers.append(multiplier)
            
            # Extract any DataFrame-compatible GIS features
            # (excludes nested dicts and complex objects)
            for key, value in enhanced.items():
                if key not in df.columns and not isinstance(value, (dict, list)):
                    df.at[idx, key] = value
        
        # Add multiplier column
        df['gis_price_multiplier'] = gis_multipliers
        
        # Calculate adjusted values if price column exists
        price_columns = ['list_price', 'estimated_value', 'last_sale_price']
        for col in price_columns:
            if col in df.columns:
                df[f'{col}_gis_adjusted'] = df[col] * df['gis_price_multiplier']
                
        logger.info("GIS enhancement complete")
        return df
        
    except Exception as e:
        logger.error(f"Error in batch GIS processing: {str(e)}", exc_info=True)
        return properties_df  # Return original DataFrame on error