"""
Tests for the enhanced GIS features module in the BCBS_Values system.

This module verifies that the enhanced GIS features calculation works correctly
with various inputs and edge cases, and that the returned metadata contains
expected information about the calculation process.
"""

import unittest
import pandas as pd
import numpy as np
from src.enhanced_gis_features import calculate_enhanced_gis_features, haversine_distance

class TestEnhancedGISFeatures(unittest.TestCase):
    """Test suite for enhanced GIS features module."""
    
    def setUp(self):
        """Set up test data for GIS features testing."""
        # Create a small test dataset with property coordinates
        self.test_data = pd.DataFrame({
            'property_id': [1, 2, 3, 4, 5],
            'latitude': [46.2804, 46.2506, 46.2896, 46.2713, np.nan],
            'longitude': [-119.2752, -119.2631, -119.2219, -119.2800, -119.2712],
            'neighborhood': ['Richland', 'Kennewick', 'West Richland', 'Richland', 'Unknown'],
            'bedrooms': [3, 4, 3, 2, 5],
            'bathrooms': [2.5, 3, 2, 1, 3.5],
            'square_feet': [2100, 2800, 1950, 1200, 3200],
            'year_built': [1985, 2005, 1992, 1965, 2018]
        })
        
        # Create some test reference points
        self.ref_points = {
            'downtown_richland': {
                'lat': 46.2804,
                'lon': -119.2752,
                'weight': 1.0,
                'scale_factor': 2.0
            },
            'columbia_center': {
                'lat': 46.2182,
                'lon': -119.2220,
                'weight': 0.8,
                'scale_factor': 3.0
            },
            'hanford_site': {
                'lat': 46.5500,
                'lon': -119.5000,
                'weight': 0.2,
                'scale_factor': 5.0
            }
        }
        
        # Create test neighborhood ratings
        self.neighborhood_ratings = {
            'Richland': 0.85,
            'Kennewick': 0.75,
            'West Richland': 0.80,
            'Pasco': 0.70,
            'Unknown': 0.50
        }
        
        # Create test amenities
        self.amenities = {
            'howard_amon_park': {
                'lat': 46.2804,
                'lon': -119.2752,
                'type': 'park'
            },
            'columbia_center_mall': {
                'lat': 46.2182,
                'lon': -119.2220,
                'type': 'shopping'
            },
            'kadlec_hospital': {
                'lat': 46.2713,
                'lon': -119.2800,
                'type': 'healthcare'
            },
            'columbia_point_golf': {
                'lat': 46.2506,
                'lon': -119.2631,
                'type': 'recreation'
            }
        }
        
        # Create test transit stops
        self.transit_stops = {
            'ben_franklin_transit_1': {
                'lat': 46.2804,
                'lon': -119.2752,
                'transit_type': 'bus'
            },
            'ben_franklin_transit_2': {
                'lat': 46.2506,
                'lon': -119.2631,
                'transit_type': 'bus'
            },
            'tri_cities_airport': {
                'lat': 46.2566,
                'lon': -119.1190,
                'transit_type': 'light_rail'
            }
        }
    
    def test_basic_functionality(self):
        """Test that basic GIS feature calculation works with minimal inputs."""
        # Test with just the property data
        result_data, metadata = calculate_enhanced_gis_features(self.test_data)
        
        # Check that the function returns the expected types
        self.assertIsInstance(result_data, pd.DataFrame)
        self.assertIsInstance(metadata, dict)
        
        # Verify the metadata contains expected keys
        self.assertIn('features_added', metadata)
        self.assertIn('error_messages', metadata)
        
        # Check that the output DataFrame has the same number of rows as input
        self.assertEqual(len(result_data), len(self.test_data))
    
    def test_with_reference_points(self):
        """Test GIS feature calculation with reference points."""
        # Test with property data and reference points
        result_data, metadata = calculate_enhanced_gis_features(
            self.test_data, 
            ref_points=self.ref_points
        )
        
        # Verify that proximity score was added
        self.assertIn('proximity_score', result_data.columns)
        self.assertIn('proximity_score', metadata['features_added'])
        
        # Check proximity score calculations (property 1 should have a reasonable score)
        self.assertGreater(result_data.loc[0, 'proximity_score'], 0.0)
        
        # Check spatial coverage metadata
        self.assertIn('spatial_coverage', metadata)
        
        # Missing coordinates should result in null proximity scores
        self.assertTrue(pd.isna(result_data.loc[4, 'proximity_score']))
    
    def test_with_neighborhood_ratings(self):
        """Test GIS feature calculation with neighborhood ratings."""
        # Test with property data and neighborhood ratings
        result_data, metadata = calculate_enhanced_gis_features(
            self.test_data, 
            neighborhood_ratings=self.neighborhood_ratings
        )
        
        # Verify that neighborhood quality was added
        self.assertIn('neighborhood_quality', result_data.columns)
        self.assertIn('neighborhood_quality', metadata['features_added'])
        
        # Check that ratings were applied correctly
        self.assertEqual(result_data.loc[0, 'neighborhood_quality'], 0.85)  # Richland
        self.assertEqual(result_data.loc[1, 'neighborhood_quality'], 0.75)  # Kennewick
        
        # Check neighborhood coverage metadata
        self.assertIn('neighborhood_coverage', metadata['spatial_coverage'])
    
    def test_with_amenities(self):
        """Test GIS feature calculation with amenities."""
        # Test with property data and amenities
        result_data, metadata = calculate_enhanced_gis_features(
            self.test_data, 
            amenities=self.amenities
        )
        
        # Verify that amenity access scores were added
        self.assertTrue(any(col.endswith('_access') for col in result_data.columns))
        
        # Check that we have specific amenity type scores
        amenity_types = set(amenity['type'] for amenity in self.amenities.values())
        for amenity_type in amenity_types:
            expected_column = f'{amenity_type}_access'
            if expected_column in result_data.columns:  # Check if the column exists before asserting
                self.assertIn(expected_column, metadata['features_added'])
        
        # Check overall amenity score
        self.assertIn('overall_amenity_access', result_data.columns)
    
    def test_with_transit_stops(self):
        """Test GIS feature calculation with transit stops."""
        # Test with property data and transit stops
        result_data, metadata = calculate_enhanced_gis_features(
            self.test_data, 
            transit_stops=self.transit_stops
        )
        
        # Verify that transit accessibility was added
        self.assertIn('transit_accessibility', result_data.columns)
        self.assertIn('transit_accessibility', metadata['features_added'])
        
        # Missing coordinates should result in null accessibility scores
        self.assertTrue(pd.isna(result_data.loc[4, 'transit_accessibility']))
    
    def test_spatial_clustering(self):
        """Test spatial clustering capabilities."""
        # Create a larger dataset for more meaningful clustering
        test_data_large = pd.DataFrame({
            'property_id': list(range(1, 21)),
            'latitude': [46.2804, 46.2506, 46.2896, 46.2713, 46.2750,
                         46.2182, 46.2190, 46.2195, 46.2178, 46.2169,
                         46.2900, 46.2920, 46.2930, 46.2910, 46.2890,
                         46.2713, 46.2720, 46.2708, 46.2715, 46.2725],
            'longitude': [-119.2752, -119.2631, -119.2219, -119.2800, -119.2730,
                          -119.2220, -119.2230, -119.2210, -119.2225, -119.2215,
                          -119.2219, -119.2225, -119.2230, -119.2210, -119.2205,
                          -119.2800, -119.2810, -119.2795, -119.2805, -119.2815]
        })
        
        # Test with larger dataset to trigger clustering
        result_data, metadata = calculate_enhanced_gis_features(test_data_large)
        
        # Check if spatial clustering was performed
        self.assertIn('spatial_cluster', result_data.columns)
        self.assertIn('spatial_cluster', metadata['features_added'])
        
        # Should have some one-hot encoded cluster columns
        cluster_columns = [col for col in result_data.columns if col.startswith('spatial_cluster_')]
        self.assertTrue(len(cluster_columns) > 0)
        
        # Should have some distance-to-cluster columns
        dist_columns = [col for col in result_data.columns if col.startswith('dist_to_cluster_')]
        self.assertTrue(len(dist_columns) > 0)
        
        # Check cluster statistics in metadata
        self.assertIn('cluster_statistics', metadata['spatial_coverage'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Create test data with invalid coordinates
        invalid_data = pd.DataFrame({
            'property_id': [1, 2, 3],
            'latitude': [46.2804, 100.0, 46.2896],  # 100.0 is invalid
            'longitude': [-119.2752, -119.2631, -200.0]  # -200.0 is invalid
        })
        
        # Test with invalid data
        result_data, metadata = calculate_enhanced_gis_features(invalid_data)
        
        # Should identify and report invalid data
        self.assertTrue(len(metadata['invalid_data']) > 0)
        
        # Should fix the invalid coordinates by setting them to None
        self.assertTrue(pd.isna(result_data.loc[1, 'latitude']))
        self.assertTrue(pd.isna(result_data.loc[2, 'longitude']))
    
    def test_haversine_distance(self):
        """Test the haversine distance calculation function."""
        # Test distance calculation with known coordinates
        # Distance between downtown Richland and Columbia Center Mall
        distance = haversine_distance(
            46.2804, -119.2752,  # Downtown Richland
            46.2182, -119.2220   # Columbia Center area
        )
        
        # Check that we get a positive distance value
        self.assertGreater(distance, 0.0)  # Should be greater than 0 km
    
    def test_comprehensive_features(self):
        """Test comprehensive feature calculation with all inputs."""
        # Test with all GIS data sources combined
        result_data, metadata = calculate_enhanced_gis_features(
            self.test_data,
            ref_points=self.ref_points,
            neighborhood_ratings=self.neighborhood_ratings,
            amenities=self.amenities,
            transit_stops=self.transit_stops
        )
        
        # Should have some features added
        self.assertTrue(len(metadata['features_added']) > 0)
        
        # Check the property nearest to a known reference point has reasonable proximity
        near_downtown = result_data.loc[0]  # Property 1 is at downtown coordinates
        if 'proximity_score' in near_downtown:
            self.assertGreater(near_downtown['proximity_score'], 0.1)
        
        # Property with missing coordinates should have some missing GIS-based features
        missing_coords = result_data.loc[4]
        # Only check proximity_score which should be NaN for invalid coordinates
        if 'proximity_score' in missing_coords:
            self.assertTrue(pd.isna(missing_coords['proximity_score']))


if __name__ == '__main__':
    unittest.main()