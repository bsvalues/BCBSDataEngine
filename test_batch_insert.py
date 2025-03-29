"""
Test script for the batch_insert_properties function.
This script demonstrates how to use the batch insertion functionality.
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
import random
from db.database import Database

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_properties(num_records=100):
    """
    Create a test DataFrame with property data.
    
    Args:
        num_records (int): Number of test records to create
        
    Returns:
        pd.DataFrame: DataFrame containing test property data
    """
    logger.info(f"Creating {num_records} test property records")
    
    # Sample property types
    property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Land']
    
    # Sample cities
    cities = ['Springfield', 'Riverdale', 'Oakwood', 'Lakeside', 'Mountainview']
    
    # Sample states
    states = ['CA', 'TX', 'FL', 'NY', 'IL']
    
    # Generate test data
    data = []
    for i in range(num_records):
        # Generate a unique property ID
        property_id = f"PROP-{i+1:06d}"
        
        # Random property characteristics
        bedrooms = random.randint(1, 6)
        bathrooms = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        square_feet = random.randint(800, 5000)
        lot_size = random.randint(1000, 20000)
        year_built = random.randint(1950, 2024)
        
        # Random location
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = f"{random.randint(10000, 99999)}"
        address = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Pine'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr', 'Rd'])}"
        
        # Random price and valuation
        list_price = random.randint(100000, 2000000)
        estimated_value = int(list_price * random.uniform(0.9, 1.1))
        last_sale_price = int(list_price * random.uniform(0.8, 0.95))
        last_sale_date = datetime.now() - timedelta(days=random.randint(30, 730))
        
        # Create property record
        property_record = {
            "property_id": property_id,
            "parcel_id": f"PARCEL-{i+1:06d}",
            "mls_id": f"MLS-{i+1:06d}",
            "address": address,
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "property_type": random.choice(property_types),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "square_feet": square_feet,
            "lot_size": lot_size,
            "year_built": year_built,
            "list_price": list_price,
            "estimated_value": estimated_value,
            "last_sale_price": last_sale_price,
            "last_sale_date": last_sale_date,
            "data_source": "TEST",
            "import_date": datetime.now()
        }
        
        data.append(property_record)
    
    # Create DataFrame from the data
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} records and {len(df.columns)} columns")
    return df

def main():
    """Test the batch insert functionality with test property data."""
    logger.info("Starting batch insert test")
    
    try:
        # Create a Database instance
        db = Database()
        
        # Generate test property data (100 records by default)
        properties_df = create_test_properties()
        
        # Insert test data in batches (batch_size=25 for demonstration)
        records_inserted = db.batch_insert_properties(properties_df, batch_size=25)
        logger.info(f"Successfully inserted {records_inserted} records")
        
        # Create indexes on the properties table
        db.create_properties_indexes()
        
        # Close the database connection
        db.close()
        
        logger.info("Batch insert test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in batch insert test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()