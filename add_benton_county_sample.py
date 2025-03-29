"""
Script to add sample Benton County, WA property data.
This ensures we have at least some data to display when focusing on Benton County.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from db.database import Database
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_benton_county_samples():
    """
    Create sample property data for Benton County, WA.
    This is used to ensure we have some data to show when focusing on this region.
    """
    # Sample property data for Benton County cities (Richland, Kennewick, Prosser, West Richland)
    sample_properties = [
        {
            "address": "1234 Columbia Ave",
            "city": "Richland",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99352",
            "property_type": "Single Family",
            "bedrooms": 4,
            "bathrooms": 2.5,
            "square_feet": 2400,
            "lot_size": 10000,
            "year_built": 1995,
            "list_price": 425000,
            "estimated_value": 432000,
            "data_source": "PACS",
            "import_date": datetime.now()
        },
        {
            "address": "5678 Gage Blvd",
            "city": "Kennewick",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99336",
            "property_type": "Single Family",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 1850,
            "lot_size": 8500,
            "year_built": 1988,
            "list_price": 375000,
            "estimated_value": 382000,
            "data_source": "PACS",
            "import_date": datetime.now()
        },
        {
            "address": "910 Wine Country Rd",
            "city": "Prosser",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99350",
            "property_type": "Single Family",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 1750,
            "lot_size": 7500,
            "year_built": 1978,
            "list_price": 325000,
            "estimated_value": 332000,
            "data_source": "PACS",
            "import_date": datetime.now()
        },
        {
            "address": "2468 Bombing Range Rd",
            "city": "West Richland",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99353",
            "property_type": "Single Family",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2800,
            "lot_size": 12000,
            "year_built": 2005,
            "list_price": 475000,
            "estimated_value": 482000,
            "data_source": "PACS",
            "import_date": datetime.now()
        },
        {
            "address": "1357 Keene Rd",
            "city": "Richland",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99352",
            "property_type": "Condo",
            "bedrooms": 2,
            "bathrooms": 2,
            "square_feet": 1250,
            "lot_size": 0,
            "year_built": 2010,
            "list_price": 245000,
            "estimated_value": 252000,
            "data_source": "MLS",
            "import_date": datetime.now()
        },
        {
            "address": "9876 Clearwater Ave",
            "city": "Kennewick",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99336",
            "property_type": "Townhome",
            "bedrooms": 3,
            "bathrooms": 2.5,
            "square_feet": 1650,
            "lot_size": 2500,
            "year_built": 2015,
            "list_price": 295000,
            "estimated_value": 302000,
            "data_source": "MLS",
            "import_date": datetime.now()
        },
        {
            "address": "5432 Horse Heaven Hills Rd",
            "city": "Prosser",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99350",
            "property_type": "Land",
            "bedrooms": 0,
            "bathrooms": 0,
            "square_feet": 0,
            "lot_size": 217800, # 5 acres
            "year_built": 0,
            "list_price": 145000,
            "estimated_value": 148000,
            "data_source": "NARRPR",
            "import_date": datetime.now()
        },
        {
            "address": "1122 Paradise Way",
            "city": "West Richland",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99353",
            "property_type": "Single Family",
            "bedrooms": 5,
            "bathrooms": 3.5,
            "square_feet": 3200,
            "lot_size": 15000,
            "year_built": 2018,
            "list_price": 525000,
            "estimated_value": 534000,
            "data_source": "MLS",
            "import_date": datetime.now()
        },
        {
            "address": "3344 Stevens Dr",
            "city": "Richland",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99352",
            "property_type": "Commercial",
            "bedrooms": 0,
            "bathrooms": 2,
            "square_feet": 3500,
            "lot_size": 20000,
            "year_built": 1972,
            "list_price": 575000,
            "estimated_value": 582000,
            "data_source": "NARRPR",
            "import_date": datetime.now()
        },
        {
            "address": "7788 Columbia Center Blvd",
            "city": "Kennewick",
            "county": "Benton",
            "state": "WA",
            "zip_code": "99336",
            "property_type": "Multi-Family",
            "bedrooms": 8,
            "bathrooms": 4,
            "square_feet": 3800,
            "lot_size": 12000,
            "year_built": 1985,
            "list_price": 625000,
            "estimated_value": 632000,
            "data_source": "MLS",
            "import_date": datetime.now()
        }
    ]
    
    return pd.DataFrame(sample_properties)

def main():
    """Check if we have Benton County, WA data and add sample data if needed."""
    try:
        # Connect to database
        db = Database()
        
        # Check if we have any Benton County properties
        query = "SELECT COUNT(*) FROM properties WHERE county = 'Benton' AND state = 'WA'"
        result = pd.read_sql(query, db.engine)
        count = result.iloc[0, 0]
        
        logger.info(f"Found {count} existing Benton County, WA properties")
        
        if count == 0:
            # No Benton County data, add our samples
            logger.info("Adding sample Benton County, WA properties")
            sample_df = create_benton_county_samples()
            inserted = db.batch_insert_properties(sample_df)
            logger.info(f"Successfully added {inserted} sample properties")
            
            # Create indexes if needed
            db.create_properties_indexes()
        else:
            logger.info("Sample data not needed - Benton County properties already exist")
            
        db.close()
        
    except SQLAlchemyError as e:
        logger.error(f"Database error adding sample data: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()