"""
Enhanced GIS features module for the BCBS_Values system.

This module provides advanced geospatial analysis capabilities for property 
valuation, integrating GIS data with traditional property attributes to 
create more accurate and spatially-aware valuation models.
"""

import logging
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of the first point in decimal degrees
    lat2, lon2 : float
        Latitude and longitude of the second point in decimal degrees
        
    Returns:
    --------
    float
        Distance between the points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r

def calculate_enhanced_gis_features(property_data, gis_data=None, ref_points=None, 
                                   neighborhood_ratings=None, amenities=None, 
                                   transit_stops=None, crime_data=None):
    """
    Calculate enhanced GIS features for properties based on geospatial data.
    
    This function processes raw property data and enhances it with sophisticated
    GIS-derived features such as proximity scores, neighborhood quality ratings,
    amenity access scores, transit accessibility, and spatial clusters. It provides
    comprehensive error handling for missing data and returns detailed metadata
    about the calculation process.
    
    Parameters:
    -----------
    property_data : pandas.DataFrame
        Dataset containing property information including latitude and longitude coordinates.
    
    gis_data : dict or pandas.DataFrame, optional
        Additional GIS data such as flood risk, school quality, etc.
    
    ref_points : dict, optional
        Dictionary of reference points with lat/lon coordinates and weights.
    
    neighborhood_ratings : dict, optional
        Dictionary mapping neighborhoods to quality ratings.
        
    amenities : dict, optional
        Dictionary of amenity points (e.g., parks, shops, restaurants) with lat/lon
        coordinates and type classification.
        
    transit_stops : dict, optional
        Dictionary of transit stops with lat/lon coordinates and transit type.
        
    crime_data : pandas.DataFrame or dict, optional
        Crime statistics by area or coordinates.
    
    Returns:
    --------
    pandas.DataFrame
        The original property data enhanced with GIS features.
    dict
        Metadata about the GIS features including validity and error information.
    """
    logger.info("Calculating enhanced GIS features for property data")
    
    # Create a copy of the input data to avoid modifying the original
    result_data = property_data.copy()
    
    # Define placeholder for GIS features we'll add
    gis_features = []
    
    # Metadata dict to track feature calculation status and errors
    gis_metadata = {
        "features_added": [],
        "invalid_data": [],
        "error_messages": [],
        "missing_required_columns": [],
        "spatial_coverage": {}
    }
    
    # Check if we have coordinates for geospatial analysis
    has_coordinates = ('latitude' in result_data.columns and 'longitude' in result_data.columns)
    
    if not has_coordinates:
        msg = "No latitude/longitude coordinates found, GIS features will be limited"
        logger.warning(msg)
        gis_metadata["error_messages"].append(msg)
        gis_metadata["missing_required_columns"].extend(["latitude", "longitude"])
    
    # Validate coordinate data if present
    if has_coordinates:
        # Check for invalid coordinates
        invalid_lat = (~result_data['latitude'].between(-90, 90)).sum()
        invalid_lon = (~result_data['longitude'].between(-180, 180)).sum()
        
        if invalid_lat > 0 or invalid_lon > 0:
            msg = f"Found {invalid_lat} invalid latitude values and {invalid_lon} invalid longitude values"
            logger.warning(msg)
            gis_metadata["invalid_data"].append(msg)
            
            # Fix invalid coordinates
            result_data.loc[~result_data['latitude'].between(-90, 90), 'latitude'] = None
            result_data.loc[~result_data['longitude'].between(-180, 180), 'longitude'] = None
    
    # 1. Calculate proximity scores if reference points are provided
    if has_coordinates and ref_points is not None:
        logger.info(f"Calculating proximity scores using {len(ref_points)} reference points")
        
        # List to store proximity scores for each property
        proximity_scores = []
        point_distances = {point: [] for point in ref_points.keys()}
        
        # For each property, calculate proximity to each reference point
        for idx, row in result_data.iterrows():
            # Calculate proximity to each reference point
            total_score = 0
            total_weight = 0
            
            # Skip rows with missing coordinates
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                proximity_scores.append(None)
                continue
            
            try:
                for point_name, point_data in ref_points.items():
                    # Skip points with missing data
                    if 'lat' not in point_data or 'lon' not in point_data or 'weight' not in point_data:
                        if point_name not in gis_metadata["invalid_data"]:
                            msg = f"Reference point '{point_name}' is missing required data"
                            logger.warning(msg)
                            gis_metadata["invalid_data"].append(msg)
                        continue
                    
                    # Calculate distance in kilometers
                    point_lat = point_data['lat']
                    point_lon = point_data['lon']
                    weight = point_data['weight']
                    
                    # Validate reference point coordinates
                    if not (-90 <= point_lat <= 90) or not (-180 <= point_lon <= 180):
                        if point_name not in gis_metadata["invalid_data"]:
                            msg = f"Reference point '{point_name}' has invalid coordinates: lat={point_lat}, lon={point_lon}"
                            logger.warning(msg)
                            gis_metadata["invalid_data"].append(msg)
                        continue
                    
                    # Calculate Haversine distance
                    try:
                        distance = haversine_distance(
                            row['latitude'], row['longitude'], 
                            point_lat, point_lon
                        )
                        
                        # Store distance for later analysis
                        point_distances[point_name].append(distance)
                        
                        # Convert distance to a score (closer = higher score)
                        # Using exponential decay: score = e^(-distance/scale_factor)
                        scale_factor = point_data.get('scale_factor', 2.0)  # 2 km default scale factor
                        score = math.exp(-distance / scale_factor)
                        
                        # Apply weight to score
                        weighted_score = score * weight
                        
                        total_score += weighted_score
                        total_weight += weight
                    except Exception as e:
                        logger.debug(f"Error calculating distance to {point_name}: {e}")
                
                # Normalize the total score
                if total_weight > 0:
                    normalized_score = total_score / total_weight
                else:
                    normalized_score = 0
                
                proximity_scores.append(normalized_score)
            
            except Exception as e:
                logger.warning(f"Error calculating proximity score for index {idx}: {e}")
                proximity_scores.append(None)  # Use None rather than 0 to indicate error
        
        # Add proximity scores to the dataset
        result_data['proximity_score'] = proximity_scores
        gis_features.append('proximity_score')
        gis_metadata["features_added"].append('proximity_score')
        
        # Calculate spatial coverage statistics
        for point_name, distances in point_distances.items():
            if distances:
                gis_metadata["spatial_coverage"][point_name] = {
                    "min_distance": min(distances),
                    "max_distance": max(distances),
                    "avg_distance": sum(distances) / len(distances),
                    "properties_in_range": sum(1 for d in distances if d < 10)  # Within 10km
                }
    
    # 2. Add neighborhood quality ratings if provided
    if neighborhood_ratings is not None:
        logger.info("Adding neighborhood quality ratings")
        
        neighborhood_scores = []
        neighborhood_coverage = {}
        
        # Check if we have a neighborhood column, if not try to extract from address
        if 'neighborhood' not in result_data.columns and 'address' in result_data.columns:
            # Simple extraction from address (adapt to your address format)
            neighborhoods = []
            for address in result_data['address']:
                try:
                    parts = str(address).split(',')
                    if len(parts) >= 2:
                        neighborhood = parts[1].strip()
                    else:
                        neighborhood = 'Unknown'
                except:
                    neighborhood = 'Unknown'
                neighborhoods.append(neighborhood)
            
            result_data['neighborhood'] = neighborhoods
        
        # Add neighborhood quality scores
        if 'neighborhood' in result_data.columns:
            # Check what percentage of neighborhoods we have ratings for
            unique_neighborhoods = result_data['neighborhood'].unique()
            rating_coverage = sum(1 for n in unique_neighborhoods if n in neighborhood_ratings) / len(unique_neighborhoods)
            
            # Add scores with detailed logging
            for neighborhood in result_data['neighborhood']:
                if neighborhood in neighborhood_ratings:
                    score = neighborhood_ratings[neighborhood]
                    neighborhood_scores.append(score)
                    
                    # Track coverage
                    if neighborhood not in neighborhood_coverage:
                        neighborhood_coverage[neighborhood] = 1
                    else:
                        neighborhood_coverage[neighborhood] += 1
                else:
                    # Use sophisticated default based on nearby known neighborhoods
                    if has_coordinates and 'latitude' in result_data.columns:
                        # Get nearby neighborhoods with known scores (simplified approach)
                        nearby_scores = []
                        # Implementation would depend on how your neighborhood boundaries are defined
                        # For now, using default fallback
                        
                        if nearby_scores:
                            default_score = sum(nearby_scores) / len(nearby_scores)
                        else:
                            default_score = 0.5  # Middle value as default
                    else:
                        default_score = 0.5
                    
                    neighborhood_scores.append(default_score)
            
            result_data['neighborhood_quality'] = neighborhood_scores
            gis_features.append('neighborhood_quality')
            gis_metadata["features_added"].append('neighborhood_quality')
            gis_metadata["spatial_coverage"]["neighborhood_coverage"] = rating_coverage
            
            # Add enhanced neighborhood metrics if we have enough coverage
            if rating_coverage > 0.5:  # More than half of neighborhoods have ratings
                # Calculate relative neighborhood quality (compared to surrounding areas)
                neighborhood_avg = result_data.groupby('neighborhood')['neighborhood_quality'].mean()
                global_avg = result_data['neighborhood_quality'].mean()
                
                relative_scores = []
                for idx, row in result_data.iterrows():
                    try:
                        n_avg = neighborhood_avg.get(row['neighborhood'], global_avg)
                        relative_scores.append(row['neighborhood_quality'] / max(n_avg, 0.1))
                    except:
                        relative_scores.append(1.0)
                
                result_data['relative_neighborhood_quality'] = relative_scores
                gis_features.append('relative_neighborhood_quality')
                gis_metadata["features_added"].append('relative_neighborhood_quality')
        else:
            msg = "No neighborhood data available, skipping neighborhood quality ratings"
            logger.warning(msg)
            gis_metadata["error_messages"].append(msg)
            gis_metadata["missing_required_columns"].append('neighborhood')
    
    # 3. Add amenity access scores if available
    if has_coordinates and amenities is not None:
        logger.info(f"Calculating amenity access scores using {len(amenities)} amenity points")
        
        # Prepare amenity access columns
        amenity_types = set()
        for amenity in amenities.values():
            if 'type' in amenity:
                amenity_types.add(amenity['type'])
        
        # Initialize columns for each amenity type
        for amenity_type in amenity_types:
            result_data[f'{amenity_type}_access'] = 0.0
            gis_features.append(f'{amenity_type}_access')
            gis_metadata["features_added"].append(f'{amenity_type}_access')
        
        # Calculate access scores
        for idx, row in result_data.iterrows():
            # Skip rows with missing coordinates
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
                
            try:
                # Group amenities by type
                amenity_by_type = {}
                
                for amenity_name, amenity_data in amenities.items():
                    # Skip amenities with missing data
                    if 'lat' not in amenity_data or 'lon' not in amenity_data or 'type' not in amenity_data:
                        continue
                    
                    amenity_type = amenity_data['type']
                    if amenity_type not in amenity_by_type:
                        amenity_by_type[amenity_type] = []
                    
                    amenity_by_type[amenity_type].append(amenity_data)
                
                # Calculate access score for each type (closer amenities = higher score)
                for amenity_type, amenity_list in amenity_by_type.items():
                    # Get distances to all amenities of this type
                    distances = []
                    
                    for amenity_data in amenity_list:
                        try:
                            distance = haversine_distance(
                                row['latitude'], row['longitude'],
                                amenity_data['lat'], amenity_data['lon']
                            )
                            distances.append(distance)
                        except:
                            continue
                    
                    if distances:
                        # Calculate access score based on closest amenities
                        # Focus on the 3 closest amenities of each type
                        closest = sorted(distances)[:3]
                        
                        # Use exponential decay formula
                        scale_factor = 1.0  # 1 km scale factor for amenities
                        score = sum(math.exp(-d / scale_factor) for d in closest) / len(closest)
                        
                        result_data.at[idx, f'{amenity_type}_access'] = score
            
            except Exception as e:
                logger.warning(f"Error calculating amenity access for index {idx}: {e}")
        
        # Create composite amenity score
        if amenity_types:
            amenity_columns = [f'{amenity_type}_access' for amenity_type in amenity_types]
            result_data['overall_amenity_access'] = result_data[amenity_columns].mean(axis=1)
            gis_features.append('overall_amenity_access')
            gis_metadata["features_added"].append('overall_amenity_access')
    
    # 4. Add transit accessibility if available
    if has_coordinates and transit_stops is not None:
        logger.info("Calculating transit accessibility scores")
        
        transit_scores = []
        
        for idx, row in result_data.iterrows():
            # Skip rows with missing coordinates
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                transit_scores.append(None)
                continue
                
            try:
                # Calculate distances to transit stops
                transit_distances = []
                
                for stop_name, stop_data in transit_stops.items():
                    # Skip stops with missing data
                    if 'lat' not in stop_data or 'lon' not in stop_data:
                        continue
                    
                    # Calculate distance
                    distance = haversine_distance(
                        row['latitude'], row['longitude'],
                        stop_data['lat'], stop_data['lon']
                    )
                    
                    # Apply importance weight based on transit type
                    weight = 1.0
                    if 'transit_type' in stop_data:
                        # Adjust weight by transit type (e.g., subway > bus)
                        transit_type = stop_data['transit_type']
                        if transit_type == 'subway':
                            weight = 1.5
                        elif transit_type == 'light_rail':
                            weight = 1.3
                        elif transit_type == 'bus':
                            weight = 0.8
                    
                    transit_distances.append((distance, weight))
                
                if transit_distances:
                    # Calculate transit score based on weighted proximity
                    # Focus on the 5 closest transit stops
                    closest = sorted(transit_distances, key=lambda x: x[0])[:5]
                    
                    # Use exponential decay formula with weighted importance
                    scale_factor = 0.5  # 0.5 km scale factor for transit
                    transit_score = sum(weight * math.exp(-d / scale_factor) 
                                       for d, weight in closest) / sum(weight for _, weight in closest)
                    
                    transit_scores.append(transit_score)
                else:
                    transit_scores.append(0)  # No transit stops
            
            except Exception as e:
                logger.warning(f"Error calculating transit score for index {idx}: {e}")
                transit_scores.append(None)
        
        # Add transit score to dataset
        result_data['transit_accessibility'] = transit_scores
        gis_features.append('transit_accessibility')
        gis_metadata["features_added"].append('transit_accessibility')
    
    # 5. Add crime risk assessment if available
    if crime_data is not None:
        logger.info("Adding crime risk assessment")
        
        # Implement based on how your crime data is structured
        # This is a simplified placeholder
        
        if isinstance(crime_data, dict) and 'risk_by_area' in crime_data:
            # If we have neighborhood-level risk scores
            if 'neighborhood' in result_data.columns:
                crime_risks = []
                
                for neighborhood in result_data['neighborhood']:
                    if neighborhood in crime_data['risk_by_area']:
                        crime_risks.append(crime_data['risk_by_area'][neighborhood])
                    else:
                        crime_risks.append(0.5)  # Default risk
                
                result_data['crime_risk'] = crime_risks
                gis_features.append('crime_risk')
                gis_metadata["features_added"].append('crime_risk')
        elif isinstance(crime_data, pd.DataFrame) and has_coordinates:
            # If we have coordinate-based crime data, implement spatial join logic
            # This would require more complex spatial operations
            # Implementation depends on how your crime data is structured
            logger.info("Coordinate-based crime data available but not implemented")
            gis_metadata["error_messages"].append("Coordinate-based crime analysis not implemented")
    
    # 6. Create spatial clusters if we have coordinates
    if has_coordinates:
        try:
            logger.info("Creating spatial clusters to identify geographic patterns")
            
            # Extract coordinates for clustering (ignoring NaN values)
            valid_coords = result_data[['latitude', 'longitude']].dropna()
            
            # Only proceed if we have enough valid coordinates
            if len(valid_coords) >= 5:  # Need at least a few points for meaningful clusters
                # Determine optimal number of clusters (between 2 and 10, based on data size)
                max_clusters = min(10, len(valid_coords) // 5)  # Ensure enough data per cluster
                
                if max_clusters >= 2:
                    # Use between 2 and max_clusters based on data size
                    n_clusters = max(2, min(max_clusters, int(np.sqrt(len(valid_coords)) / 2)))
                    logger.info(f"Using {n_clusters} spatial clusters based on data size")
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_results = kmeans.fit(valid_coords)
                    
                    # Create a mapping from original indices to cluster labels
                    cluster_map = {}
                    for i, (idx, _) in enumerate(valid_coords.iterrows()):
                        cluster_map[idx] = cluster_results.labels_[i]
                    
                    # Apply cluster assignments to original data
                    result_data['spatial_cluster'] = None  # Initialize with None
                    for idx in cluster_map:
                        result_data.at[idx, 'spatial_cluster'] = cluster_map[idx]
                    
                    # Convert to categorical (for later analysis)
                    if 'spatial_cluster' in result_data.columns:
                        result_data['spatial_cluster'] = result_data['spatial_cluster'].astype('category')
                        gis_features.append('spatial_cluster')
                        gis_metadata["features_added"].append('spatial_cluster')
                        
                        # Create one-hot encoding of clusters for modeling
                        for cluster in range(n_clusters):
                            col_name = f'spatial_cluster_{cluster}'
                            result_data[col_name] = (result_data['spatial_cluster'] == cluster).astype(int)
                            gis_features.append(col_name)
                            gis_metadata["features_added"].append(col_name)
                        
                        # Add distance to cluster centers
                        for cluster, center in enumerate(cluster_results.cluster_centers_):
                            center_lat, center_lon = center
                            distances = []
                            
                            for idx, row in result_data.iterrows():
                                if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                                    distances.append(None)
                                else:
                                    dist = haversine_distance(
                                        row['latitude'], row['longitude'],
                                        center_lat, center_lon
                                    )
                                    distances.append(dist)
                            
                            result_data[f'dist_to_cluster_{cluster}'] = distances
                            gis_features.append(f'dist_to_cluster_{cluster}')
                            gis_metadata["features_added"].append(f'dist_to_cluster_{cluster}')
                        
                        # Calculate cluster statistics
                        cluster_stats = {}
                        for cluster in range(n_clusters):
                            cluster_properties = result_data[result_data['spatial_cluster'] == cluster]
                            center = cluster_results.cluster_centers_[cluster]
                            
                            stats = {
                                "count": len(cluster_properties),
                                "center_lat": center[0],
                                "center_lon": center[1]
                            }
                            
                            # Calculate mean price if available
                            if 'price' in cluster_properties.columns:
                                stats["mean_price"] = cluster_properties['price'].mean()
                            
                            # Calculate mean square feet if available
                            if 'square_feet' in cluster_properties.columns:
                                stats["mean_sqft"] = cluster_properties['square_feet'].mean()
                            
                            cluster_stats[f"cluster_{cluster}"] = stats
                        
                        gis_metadata["spatial_coverage"]["cluster_statistics"] = cluster_stats
                else:
                    msg = "Not enough distinct coordinate data for meaningful spatial clustering"
                    logger.warning(msg)
                    gis_metadata["error_messages"].append(msg)
            else:
                msg = "Not enough valid coordinates for spatial clustering"
                logger.warning(msg)
                gis_metadata["error_messages"].append(msg)
                
        except Exception as e:
            msg = f"Error during spatial clustering: {e}"
            logger.warning(msg)
            gis_metadata["error_messages"].append(msg)
    
    # 7. Add flood risk data if available
    if gis_data is not None and 'flood_risk' in gis_data:
        logger.info("Adding flood risk data")
        try:
            result_data['flood_risk'] = gis_data['flood_risk']
            gis_features.append('flood_risk')
            gis_metadata["features_added"].append('flood_risk')
        except Exception as e:
            msg = f"Could not add flood risk data: {e}"
            logger.warning(msg)
            gis_metadata["error_messages"].append(msg)
    
    # 8. Add school district quality if available
    if gis_data is not None and 'school_quality' in gis_data:
        logger.info("Adding school district quality data")
        try:
            result_data['school_quality'] = gis_data['school_quality']
            gis_features.append('school_quality')
            gis_metadata["features_added"].append('school_quality')
        except Exception as e:
            msg = f"Could not add school quality data: {e}"
            logger.warning(msg)
            gis_metadata["error_messages"].append(msg)
    
    # 9. Log what GIS features were added
    if gis_features:
        logger.info(f"Added {len(gis_features)} GIS features: {', '.join(gis_features)}")
    else:
        msg = "No GIS features were added"
        logger.warning(msg)
        gis_metadata["error_messages"].append(msg)
    
    return result_data, gis_metadata