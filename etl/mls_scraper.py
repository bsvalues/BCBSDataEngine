"""
MLS (Multiple Listing Service) data extraction module.
Handles extraction of property data from MLS systems.
"""
import os
import logging
import requests
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class MLSScraper:
    """
    Scraper for extracting data from Multiple Listing Service (MLS) APIs.
    """
    
    def __init__(self, batch_size=100):
        """
        Initialize the MLS scraper.
        
        Args:
            batch_size (int): Number of records to fetch in each API call
        """
        self.api_key = os.getenv("MLS_API_KEY")
        self.base_url = "https://mls-api.example.com/v1"  # Replace with actual MLS API endpoint
        self.batch_size = batch_size
        
        if not self.api_key:
            logger.warning("MLS_API_KEY not found in environment variables")
    
    def extract(self, start_date=None, end_date=None):
        """
        Extract property data from MLS API.
        
        Args:
            start_date (str, optional): Start date for data extraction (YYYY-MM-DD)
            end_date (str, optional): End date for data extraction (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        logger.info("Extracting data from MLS")
        
        # Set default date range if not provided
        if not start_date:
            start_date = (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            # Prepare parameters for API request
            params = {
                "api_key": self.api_key,
                "start_date": start_date,
                "end_date": end_date,
                "limit": self.batch_size,
                "offset": 0
            }
            
            all_properties = []
            more_data = True
            
            # Paginate through all results
            while more_data:
                logger.debug(f"Fetching MLS data with offset {params['offset']}")
                
                # Make API request
                response = self._make_api_request("properties", params)
                
                # Process response
                if response and "properties" in response and response["properties"]:
                    properties = response["properties"]
                    all_properties.extend(properties)
                    params["offset"] += self.batch_size
                    
                    logger.debug(f"Retrieved {len(properties)} properties from MLS")
                else:
                    more_data = False
            
            logger.info(f"Successfully extracted {len(all_properties)} properties from MLS")
            
            # Convert to DataFrame
            if all_properties:
                return pd.DataFrame(all_properties)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting data from MLS: {str(e)}", exc_info=True)
            raise
    
    def _make_api_request(self, endpoint, params):
        """
        Make a request to the MLS API.
        
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
            logger.error(f"API request to MLS failed: {str(e)}")
            return None
    
    def transform_and_load(self, data, db):
        """
        Transform MLS data and load it into the database.
        
        Args:
            data (pd.DataFrame): DataFrame containing MLS property data
            db: Database connection object
            
        Returns:
            int: Number of records loaded into the database
        """
        if data.empty:
            logger.info("No MLS data to transform and load")
            return 0
            
        logger.info(f"Transforming and loading {len(data)} MLS records")
        
        try:
            # Transform data
            transformed_data = self._transform_data(data)
            
            # Load data into database
            records_loaded = self._load_data(transformed_data, db)
            
            logger.info(f"Successfully loaded {records_loaded} MLS records into database")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Error transforming and loading MLS data: {str(e)}", exc_info=True)
            raise
    
    def _transform_data(self, data):
        """
        Transform MLS data into the format required for database.
        
        Args:
            data (pd.DataFrame): Raw MLS data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Map column names to standardized schema
        column_mapping = {
            "mlsId": "mls_id",
            "listingId": "listing_id",
            "propertyType": "property_type",
            "address": "address",
            "city": "city",
            "state": "state",
            "zipCode": "zip_code",
            "price": "list_price",
            "bedrooms": "bedrooms",
            "bathrooms": "bathrooms",
            "squareFeet": "square_feet",
            "lotSize": "lot_size",
            "yearBuilt": "year_built",
            "listingDate": "listing_date",
            "status": "status"
        }
        
        # Rename columns based on mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add source column
        df["data_source"] = "MLS"
        
        # Add import timestamp
        df["import_date"] = datetime.now()
        
        # Convert data types
        if "list_price" in df.columns:
            df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce")
        
        if "square_feet" in df.columns:
            df["square_feet"] = pd.to_numeric(df["square_feet"], errors="coerce")
        
        if "lot_size" in df.columns:
            df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce")
        
        if "year_built" in df.columns:
            df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
        
        if "listing_date" in df.columns:
            df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
        
        return df
    
    def _load_data(self, data, db):
        """
        Load transformed data into the database.
        
        Args:
            data (pd.DataFrame): Transformed MLS data
            db: Database connection object
            
        Returns:
            int: Number of records loaded
        """
        # Use the database object to insert the data
        return db.insert_properties(data, source="MLS")
