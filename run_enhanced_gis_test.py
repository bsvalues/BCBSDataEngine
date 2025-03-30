"""
Script to test the enhanced GIS features with sample Benton County property data.

This script creates sample property data and GIS reference points for the
Tri-Cities area (Richland, Kennewick, and Pasco) in Benton County, WA,
then processes them with the enhanced GIS features module and outputs
the results.
"""

import pandas as pd
import numpy as np
import logging
import json
from src.enhanced_gis_features import calculate_enhanced_gis_features
from src.valuation import engineer_property_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_gis_valuation_result.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_sample_property_data():
    """Create a sample dataset of properties in the Tri-Cities area."""
    logger.info("Creating sample property data for Benton County")
    
    # Main Tri-Cities coordinates (approximate)
    richland_coords = (46.2804, -119.2752)
    kennewick_coords = (46.2113, -119.1372)
    pasco_coords = (46.2395, -119.1005)
    
    # Random property coordinates clustered around these cities
    np.random.seed(42)  # For reproducibility
    
    # Number of properties in each city
    n_richland = 15
    n_kennewick = 12
    n_pasco = 8
    n_properties = n_richland + n_kennewick + n_pasco
    
    # Generate property locations with random variations
    cities = []
    neighborhoods = []
    latitudes = []
    longitudes = []
    
    # Richland properties
    for i in range(n_richland):
        cities.append('Richland')
        # Different neighborhoods in Richland
        neighborhoods.append(np.random.choice(['North Richland', 'Central Richland', 'South Richland', 'Horn Rapids']))
        # Add some random variation to coordinates
        lat_variation = np.random.normal(0, 0.015)
        lon_variation = np.random.normal(0, 0.015)
        latitudes.append(richland_coords[0] + lat_variation)
        longitudes.append(richland_coords[1] + lon_variation)
    
    # Kennewick properties
    for i in range(n_kennewick):
        cities.append('Kennewick')
        # Different neighborhoods in Kennewick
        neighborhoods.append(np.random.choice(['Downtown Kennewick', 'Southridge', 'Creekstone', 'Canyon Lakes']))
        # Add some random variation to coordinates
        lat_variation = np.random.normal(0, 0.012)
        lon_variation = np.random.normal(0, 0.012)
        latitudes.append(kennewick_coords[0] + lat_variation)
        longitudes.append(kennewick_coords[1] + lon_variation)
    
    # Pasco properties
    for i in range(n_pasco):
        cities.append('Pasco')
        # Different neighborhoods in Pasco
        neighborhoods.append(np.random.choice(['West Pasco', 'Road 68', 'East Pasco', 'Riverview']))
        # Add some random variation to coordinates
        lat_variation = np.random.normal(0, 0.01)
        lon_variation = np.random.normal(0, 0.01)
        latitudes.append(pasco_coords[0] + lat_variation)
        longitudes.append(pasco_coords[1] + lon_variation)
    
    # Generate property attributes
    property_ids = list(range(1, n_properties + 1))
    bedrooms = np.random.randint(2, 6, n_properties)  # 2-5 bedrooms
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_properties)
    square_feet = np.random.randint(1000, 4000, n_properties)
    lot_sizes = np.random.randint(5000, 20000, n_properties)
    year_built = np.random.randint(1950, 2023, n_properties)
    
    # Some properties have pools, garages, views
    has_pool = np.random.choice([0, 1], n_properties, p=[0.85, 0.15])
    has_garage = np.random.choice([0, 1], n_properties, p=[0.2, 0.8])
    garage_spaces = np.where(has_garage == 1, np.random.randint(1, 4, n_properties), 0)
    has_view = np.random.choice([0, 1], n_properties, p=[0.7, 0.3])
    
    # Some properties have been renovated
    has_renovation = np.random.choice([0, 1], n_properties, p=[0.7, 0.3])
    year_renovated = np.where(
        has_renovation == 1,
        np.random.randint(np.maximum(year_built, 1990), 2023),
        np.nan
    )
    
    # Create DataFrame
    property_data = pd.DataFrame({
        'property_id': property_ids,
        'city': cities,
        'neighborhood': neighborhoods,
        'latitude': latitudes,
        'longitude': longitudes,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'square_feet': square_feet,
        'lot_size': lot_sizes,
        'year_built': year_built,
        'year_renovated': year_renovated,
        'has_pool': has_pool,
        'has_garage': has_garage, 
        'garage_spaces': garage_spaces,
        'has_view': has_view
    })
    
    logger.info(f"Created {len(property_data)} sample properties")
    return property_data

def create_reference_points():
    """Create reference points for GIS analysis in the Tri-Cities area."""
    logger.info("Creating reference points for Benton County")
    
    ref_points = {
        # City centers
        'richland_downtown': {
            'lat': 46.2804,
            'lon': -119.2752,
            'weight': 1.0,
            'scale_factor': 2.0
        },
        'kennewick_downtown': {
            'lat': 46.2113,
            'lon': -119.1372,
            'weight': 0.9,
            'scale_factor': 2.0
        },
        'pasco_downtown': {
            'lat': 46.2395,
            'lon': -119.1005,
            'weight': 0.8,
            'scale_factor': 2.0
        },
        
        # Major landmarks
        'columbia_center_mall': {
            'lat': 46.2182, 
            'lon': -119.2220, 
            'weight': 0.7,
            'scale_factor': 3.0
        },
        'richland_federal_building': {
            'lat': 46.2756,
            'lon': -119.2778,
            'weight': 0.5,
            'scale_factor': 1.5
        },
        'columbia_river': {
            'lat': 46.2573,
            'lon': -119.2837,
            'weight': 0.9,
            'scale_factor': 1.0
        },
        'wsutc_campus': {
            'lat': 46.2715,
            'lon': -119.2790,
            'weight': 0.6,
            'scale_factor': 2.0
        },
        'hanford_site': {
            'lat': 46.5500,
            'lon': -119.5000,
            'weight': 0.3,
            'scale_factor': 10.0
        }
    }
    
    logger.info(f"Created {len(ref_points)} reference points")
    return ref_points

def create_neighborhood_ratings():
    """Create neighborhood quality ratings for the Tri-Cities area."""
    logger.info("Creating neighborhood ratings for Benton County")
    
    neighborhood_ratings = {
        # Richland neighborhoods
        'North Richland': 0.85,
        'Central Richland': 0.80,
        'South Richland': 0.90,
        'Horn Rapids': 0.75,
        
        # Kennewick neighborhoods
        'Downtown Kennewick': 0.70,
        'Southridge': 0.85,
        'Creekstone': 0.80,
        'Canyon Lakes': 0.92,
        
        # Pasco neighborhoods
        'West Pasco': 0.80,
        'Road 68': 0.82,
        'East Pasco': 0.65,
        'Riverview': 0.75,
        
        # Cities as fallbacks
        'Richland': 0.85,
        'Kennewick': 0.80,
        'Pasco': 0.75,
        
        # Default
        'Unknown': 0.50
    }
    
    logger.info(f"Created ratings for {len(neighborhood_ratings)} neighborhoods")
    return neighborhood_ratings

def create_amenities():
    """Create amenity points for GIS analysis in the Tri-Cities area."""
    logger.info("Creating amenity data for Benton County")
    
    amenities = {
        # Parks
        'howard_amon_park': {
            'lat': 46.2710,
            'lon': -119.2725,
            'type': 'park'
        },
        'columbia_park': {
            'lat': 46.2365,
            'lon': -119.1826,
            'type': 'park'
        },
        'leslie_groves_park': {
            'lat': 46.2948,
            'lon': -119.2745,
            'type': 'park'
        },
        
        # Shopping
        'columbia_center_mall': {
            'lat': 46.2182,
            'lon': -119.2220,
            'type': 'shopping'
        },
        'richland_uptown': {
            'lat': 46.2830,
            'lon': -119.2828,
            'type': 'shopping'
        },
        
        # Healthcare
        'kadlec_hospital': {
            'lat': 46.2700,
            'lon': -119.2737,
            'type': 'healthcare'
        },
        'trios_hospital': {
            'lat': 46.1926,
            'lon': -119.1956,
            'type': 'healthcare'
        },
        
        # Schools
        'richland_high': {
            'lat': 46.2845,
            'lon': -119.2880,
            'type': 'school'
        },
        'kamiakin_high': {
            'lat': 46.2214,
            'lon': -119.1689,
            'type': 'school'
        },
        'wsutc': {
            'lat': 46.2724,
            'lon': -119.2782,
            'type': 'school'
        },
        
        # Recreation
        'columbia_point_golf': {
            'lat': 46.2483,
            'lon': -119.2557,
            'type': 'recreation'
        },
        'richland_community_center': {
            'lat': 46.2705,
            'lon': -119.2737,
            'type': 'recreation'
        }
    }
    
    logger.info(f"Created {len(amenities)} amenity points")
    return amenities

def create_transit_stops():
    """Create transit stop points for GIS analysis in the Tri-Cities area."""
    logger.info("Creating transit data for Benton County")
    
    transit_stops = {
        # Bus stops
        'knight_street_transit': {
            'lat': 46.2804,
            'lon': -119.2777,
            'transit_type': 'bus'
        },
        'jadwin_transit': {
            'lat': 46.2713,
            'lon': -119.2769,
            'transit_type': 'bus'
        },
        'kennewick_transit': {
            'lat': 46.2113,
            'lon': -119.1372,
            'transit_type': 'bus'
        },
        'pasco_transit': {
            'lat': 46.2395,
            'lon': -119.1005,
            'transit_type': 'bus'
        },
        
        # Light rail (hypothetical future system)
        'richland_light_rail': {
            'lat': 46.2756,
            'lon': -119.2778,
            'transit_type': 'light_rail'
        },
        'kennewick_light_rail': {
            'lat': 46.2113,
            'lon': -119.1372,
            'transit_type': 'light_rail'
        },
        
        # Airport
        'tri_cities_airport': {
            'lat': 46.2566,
            'lon': -119.1190,
            'transit_type': 'airport'
        }
    }
    
    logger.info(f"Created {len(transit_stops)} transit stops")
    return transit_stops

def create_crime_data():
    """Create sample crime risk data by neighborhood."""
    logger.info("Creating crime risk data for Benton County")
    
    crime_data = {
        'risk_by_area': {
            # Richland neighborhoods
            'North Richland': 0.25,
            'Central Richland': 0.35,
            'South Richland': 0.15,
            'Horn Rapids': 0.20,
            
            # Kennewick neighborhoods
            'Downtown Kennewick': 0.55,
            'Southridge': 0.25,
            'Creekstone': 0.30,
            'Canyon Lakes': 0.15,
            
            # Pasco neighborhoods
            'West Pasco': 0.30,
            'Road 68': 0.25,
            'East Pasco': 0.60,
            'Riverview': 0.35,
            
            # Cities as fallbacks
            'Richland': 0.25,
            'Kennewick': 0.35,
            'Pasco': 0.40,
            
            # Default
            'Unknown': 0.50
        }
    }
    
    logger.info(f"Created crime risk data for {len(crime_data['risk_by_area'])} areas")
    return crime_data

def main():
    """Run enhanced GIS features test with sample data."""
    logger.info("Starting enhanced GIS features test")
    
    # Create sample data
    property_data = create_sample_property_data()
    ref_points = create_reference_points()
    neighborhood_ratings = create_neighborhood_ratings()
    amenities = create_amenities()
    transit_stops = create_transit_stops()
    crime_data = create_crime_data()
    
    # Engineer property features first
    logger.info("Engineering property features")
    property_data = engineer_property_features(property_data)
    
    # Run enhanced GIS features calculation
    logger.info("Calculating enhanced GIS features")
    result_data, metadata = calculate_enhanced_gis_features(
        property_data,
        ref_points=ref_points,
        neighborhood_ratings=neighborhood_ratings,
        amenities=amenities,
        transit_stops=transit_stops,
        crime_data=crime_data
    )
    
    # Log the results
    logger.info(f"Enhanced data contains {len(result_data.columns)} columns")
    logger.info(f"Added {len(metadata['features_added'])} GIS features: {', '.join(metadata['features_added'])}")
    
    if metadata['error_messages']:
        logger.warning(f"Encountered {len(metadata['error_messages'])} issues:")
        for msg in metadata['error_messages']:
            logger.warning(f"  - {msg}")
    
    # Create summary stats for each GIS feature added
    logger.info("Feature statistics:")
    for feature in metadata['features_added']:
        if feature in result_data.columns:
            try:
                stats = result_data[feature].describe()
                logger.info(f"  {feature}:")
                logger.info(f"    Count: {stats['count']}")
                logger.info(f"    Mean: {stats['mean']:.4f}")
                logger.info(f"    Min: {stats['min']:.4f}")
                logger.info(f"    Max: {stats['max']:.4f}")
            except:
                logger.info(f"  {feature}: Could not calculate statistics")
    
    # Save metadata to a JSON file
    with open('gis_features_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved GIS features metadata to gis_features_metadata.json")
    
    # Save a sample of the enriched data
    sample_columns = ['property_id', 'city', 'neighborhood', 'bedrooms', 'bathrooms', 'square_feet']
    sample_columns.extend(metadata['features_added'][:10])  # Add up to 10 GIS features
    
    # Ensure all columns exist in the result_data
    sample_columns = [col for col in sample_columns if col in result_data.columns]
    
    # Save sample to CSV
    sample_data = result_data[sample_columns].head(10)
    sample_data.to_csv('enhanced_gis_sample_data.csv', index=False)
    logger.info("Saved sample of enhanced data to enhanced_gis_sample_data.csv")
    
    logger.info("Enhanced GIS features test completed successfully")

if __name__ == "__main__":
    main()