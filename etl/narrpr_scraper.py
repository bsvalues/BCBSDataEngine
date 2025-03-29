"""
NARRPR (National Association of REALTORS® Realtors Property Resource) data extraction module.
Handles extraction of property data from NARRPR API.
"""
import os
import logging
import requests
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class NARRPRScraper:
    """
    Scraper for extracting data from NARRPR (National Association of REALTORS® Realtors Property Resource).
    """
    
    def __init__(self, batch_size=100):
        """
        Initialize the NARRPR scraper.
        
        Args:
            batch_size (int): Number of records to fetch in each API call
        """
        self.api_key = os.getenv("NARRPR_API_KEY")
        self.base_url = "https://api.narrpr.com/v1"  # Replace with actual NARRPR API endpoint
        self.batch_size = batch_size
        
        if not self.api_key:
            logger.warning("NARRPR_API_KEY not found in environment variables")
    
    def extract(self, location=None, property_type=None):
        """
        Extract property data from NARRPR API.
        
        Args:
            location (str, optional): Location (city, zip code) to search for properties
            property_type (str, optional): Type of property to search for
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        logger.info("Extracting data from NARRPR")
        
        try:
            # Prepare parameters for API request
            params = {
                "api_key": self.api_key,
                "limit": self.batch_size,
                "offset": 0
            }
            
            # Add optional parameters if provided
            if location:
                params["location"] = location
            
            if property_type:
                params["propertyType"] = property_type
            
            all_properties = []
            more_data = True
            
            # Paginate through all results
            while more_data:
                logger.debug(f"Fetching NARRPR data with offset {params['offset']}")
                
                # Make API request
                response = self._make_api_request("properties/search", params)
                
                # Process response
                if response and "properties" in response and response["properties"]:
                    properties = response["properties"]
                    all_properties.extend(properties)
                    params["offset"] += self.batch_size
                    
                    logger.debug(f"Retrieved {len(properties)} properties from NARRPR")
                else:
                    more_data = False
            
            logger.info(f"Successfully extracted {len(all_properties)} properties from NARRPR")
            
            # Convert to DataFrame
            if all_properties:
                return pd.DataFrame(all_properties)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting data from NARRPR: {str(e)}", exc_info=True)
            raise
    
    def _make_api_request(self, endpoint, params):
        """
        Make a request to the NARRPR API.
        
        Args:
            endpoint (str): API endpoint to call
            params (dict): Parameters for the API request
            
        Returns:
            dict: JSON response from the API
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request to NARRPR failed: {str(e)}")
            return None
    
    def transform_and_load(self, data, db):
        """
        Transform NARRPR data and load it into the database.
        
        Args:
            data (pd.DataFrame): DataFrame containing NARRPR property data
            db: Database connection object
            
        Returns:
            int: Number of records loaded into the database
        """
        if data.empty:
            logger.info("No NARRPR data to transform and load")
            return 0
            
        logger.info(f"Transforming and loading {len(data)} NARRPR records")
        
        try:
            # Transform data
            transformed_data = self._transform_data(data)
            
            # Load data into database
            records_loaded = self._load_data(transformed_data, db)
            
            logger.info(f"Successfully loaded {records_loaded} NARRPR records into database")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Error transforming and loading NARRPR data: {str(e)}", exc_info=True)
            raise
    
    def _transform_data(self, data):
        """
        Transform NARRPR data into the format required for database.
        
        Args:
            data (pd.DataFrame): Raw NARRPR data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Map column names to standardized schema
        column_mapping = {
            "propertyId": "property_id",
            "addressLine1": "address",
            "city": "city",
            "state": "state",
            "zipCode": "zip_code",
            "propertyType": "property_type",
            "estimatedValue": "estimated_value",
            "lastSalePrice": "last_sale_price",
            "lastSaleDate": "last_sale_date",
            "bedrooms": "bedrooms",
            "bathrooms": "bathrooms",
            "totalRooms": "total_rooms",
            "squareFeet": "square_feet",
            "lotSize": "lot_size",
            "yearBuilt": "year_built"
        }
        
        # Rename columns based on mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add source column
        df["data_source"] = "NARRPR"
        
        # Add import timestamp
        df["import_date"] = datetime.now()
        
        # Convert data types
        if "estimated_value" in df.columns:
            df["estimated_value"] = pd.to_numeric(df["estimated_value"], errors="coerce")
        
        if "last_sale_price" in df.columns:
            df["last_sale_price"] = pd.to_numeric(df["last_sale_price"], errors="coerce")
        
        if "last_sale_date" in df.columns:
            df["last_sale_date"] = pd.to_datetime(df["last_sale_date"], errors="coerce")
        
        if "square_feet" in df.columns:
            df["square_feet"] = pd.to_numeric(df["square_feet"], errors="coerce")
        
        if "lot_size" in df.columns:
            df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce")
        
        if "year_built" in df.columns:
            df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
        
        return df
    
    def _load_data(self, data, db):
        """
        Load transformed data into the database.
        
        Args:
            data (pd.DataFrame): Transformed NARRPR data
            db: Database connection object
            
        Returns:
            int: Number of records loaded
        """
        # Use the database object to insert the data
        return db.insert_properties(data, source="NARRPR")
