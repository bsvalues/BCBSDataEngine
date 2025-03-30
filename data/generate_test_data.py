#!/usr/bin/env python3
"""
Script to generate test property data for the BCBS_Values system.
This creates a sample CSV file with properties in Benton County, WA
that can be used for testing the valuation API.
"""
import os
import csv
import random
import numpy as np
from datetime import datetime, timedelta

def generate_test_data(num_properties=50, output_file="data/test_properties.csv"):
    """
    Generate test property data with realistic values for Benton County, WA.
    
    Args:
        num_properties (int): Number of test properties to generate
        output_file (str): Path to output CSV file
    """
    # Make sure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Define possible values for categorical fields
    cities = ["Richland", "Kennewick", "Pasco", "West Richland", "Benton City", "Prosser"]
    city_weights = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]  # Probability weights for cities
    
    neighborhoods = {
        "Richland": ["Meadow Springs", "Horn Rapids", "Central Richland", "North Richland", "South Richland"],
        "Kennewick": ["Southridge", "Creekstone", "Canyon Lakes", "Downtown Kennewick", "Zintel Canyon"],
        "Pasco": ["Road 68", "Riverview", "Downtown Pasco", "East Pasco", "West Pasco"],
        "West Richland": ["Queensgate", "Paradise South", "Flat Top", "West Richland Heights"],
        "Benton City": ["Downtown Benton City", "Red Mountain"],
        "Prosser": ["Downtown Prosser", "Wine Valley", "Prosser Heights"]
    }
    
    property_types = ["Single Family", "Townhouse", "Condo", "Multi-Family"]
    property_type_weights = [0.8, 0.1, 0.05, 0.05]  # Probability weights
    
    # Define realistic value ranges for numeric fields
    # Format: (min, max, mean, std_dev)
    sq_ft_params = (800, 4500, 2200, 800)  # Square footage 
    bedroom_options = [1, 2, 3, 4, 5, 6]
    bedroom_weights = [0.05, 0.1, 0.4, 0.3, 0.1, 0.05]  # Probability weights
    bathroom_options = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    bathroom_weights = [0.05, 0.1, 0.3, 0.3, 0.15, 0.05, 0.05]  # Probability weights
    year_built_params = (1950, 2025, 1995, 15)  # Year built
    lot_size_params = (3000, 20000, 8500, 3000)  # Lot size in sq ft
    
    # Value estimation parameters (for generating realistic property values)
    # Base price per sq ft ranges from $150-$250/sq ft in Benton County
    base_price_per_sqft_params = (150, 250, 200, 20)
    
    # Geographic bounds for Benton County, WA (approximate)
    lat_bounds = (46.0, 46.4)  # Latitude range
    lon_bounds = (-119.7, -119.0)  # Longitude range
    
    # Generate properties
    properties = []
    
    for i in range(num_properties):
        # Generate property ID
        property_id = f"BCBS-{random.randint(10000, 99999)}"
        
        # Select city and neighborhood
        city = random.choices(cities, weights=city_weights)[0]
        neighborhood = random.choice(neighborhoods[city])
        
        # Generate property type
        property_type = random.choices(property_types, weights=property_type_weights)[0]
        
        # Generate numeric features with realistic distributions
        square_feet = int(np.random.normal(sq_ft_params[2], sq_ft_params[3]))
        square_feet = max(sq_ft_params[0], min(sq_ft_params[1], square_feet))  # Clamp within bounds
        
        bedrooms = random.choices(bedroom_options, weights=bedroom_weights)[0]
        bathrooms = random.choices(bathroom_options, weights=bathroom_weights)[0]
        
        year_built = int(np.random.normal(year_built_params[2], year_built_params[3]))
        year_built = max(year_built_params[0], min(year_built_params[1], year_built))  # Clamp within bounds
        
        lot_size = int(np.random.normal(lot_size_params[2], lot_size_params[3]))
        lot_size = max(lot_size_params[0], min(lot_size_params[1], lot_size))  # Clamp within bounds
        
        # Generate latitude and longitude within Benton County bounds
        latitude = random.uniform(lat_bounds[0], lat_bounds[1])
        longitude = random.uniform(lon_bounds[0], lon_bounds[1])
        
        # Calculate property value based on features
        base_price_per_sqft = np.random.normal(base_price_per_sqft_params[2], base_price_per_sqft_params[3])
        base_price_per_sqft = max(base_price_per_sqft_params[0], min(base_price_per_sqft_params[1], base_price_per_sqft))
        
        # Adjust base price by neighborhood quality
        neighborhood_factors = {
            "Meadow Springs": 1.3, "Horn Rapids": 1.2, "Southridge": 1.2, "Canyon Lakes": 1.25,
            "Road 68": 1.1, "Queensgate": 1.15, "Wine Valley": 1.05,
            # Default factor of 1.0 for unlisted neighborhoods
        }
        
        neighborhood_factor = neighborhood_factors.get(neighborhood, 1.0)
        
        # Age discount (newer = better)
        age = 2025 - year_built
        age_factor = max(0.7, 1.0 - (age / 100))
        
        # Calculate property value
        base_value = square_feet * base_price_per_sqft
        property_value = base_value * neighborhood_factor * age_factor
        
        # Add small random variation
        property_value = property_value * random.uniform(0.9, 1.1)
        property_value = round(property_value, 2)
        
        # Generate random address
        street_number = random.randint(100, 9999)
        streets = ["Main St", "Oak Ave", "Maple Dr", "Washington St", "Columbia Blvd", "River Rd", 
                  "Canyon Dr", "Valley View", "Mountain Rd", "Park Place", "Sunset Dr", "Lake View"]
        street = random.choice(streets)
        address = f"{street_number} {street}"
        
        # Generate random timestamps within the last year
        days_ago = random.randint(1, 365)
        timestamp = datetime.now() - timedelta(days=days_ago)
        last_updated = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Assemble the property record
        property_record = {
            "property_id": property_id,
            "address": address,
            "city": city,
            "state": "WA",
            "zip_code": random.choice(["99352", "99301", "99336", "99353", "99320", "99350"]),
            "neighborhood": neighborhood,
            "property_type": property_type,
            "square_feet": square_feet,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "lot_size": lot_size,
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "list_price": property_value,  # This is the column needed for valuation
            "last_sale_price": property_value * random.uniform(0.9, 1.0),
            "last_sale_date": (timestamp - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
            "last_updated": last_updated
        }
        
        properties.append(property_record)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        if properties:
            fieldnames = properties[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for prop in properties:
                writer.writerow(prop)
    
    print(f"Generated {num_properties} test properties and saved to {output_file}")
    return output_file

if __name__ == "__main__":
    generate_test_data(num_properties=50)