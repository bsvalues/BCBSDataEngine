"""
PACS (Property Appraiser/County Assessor) data import module.
Handles extraction of property data from county assessor systems.
"""
import os
import logging
import requests
import pandas as pd
from datetime import datetime
import io

logger = logging.getLogger(__name__)

class PACSImporter:
    """
    Importer for extracting data from Property Appraiser/County Assessor Systems (PACS).
    Supports both API and file-based imports.
    """
    
    def __init__(self, batch_size=100):
        """
        Initialize the PACS importer.
        
        Args:
            batch_size (int): Number of records to process in each batch
        """
        self.api_key = os.getenv("PACS_API_KEY")
        self.base_url = "https://pacs-api.example.com/v1"  # Replace with actual PACS API endpoint
        self.batch_size = batch_size
        
        if not self.api_key:
            logger.warning("PACS_API_KEY not found in environment variables")
    
    def extract(self, county=None, state=None, file_path=None):
        """
        Extract property data from PACS API or data file.
        
        Args:
            county (str, optional): County name to search for properties
            state (str, optional): State abbreviation
            file_path (str, optional): Path to local file containing PACS data
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        logger.info("Extracting data from PACS")
        
        try:
            # If file path is provided, use file-based import
            if file_path:
                return self._extract_from_file(file_path)
            # Otherwise use API-based import
            else:
                return self._extract_from_api(county, state)
                
        except Exception as e:
            logger.error(f"Error extracting data from PACS: {str(e)}", exc_info=True)
            raise
    
    def _extract_from_api(self, county=None, state=None):
        """
        Extract property data from PACS API.
        
        Args:
            county (str, optional): County name to search for properties
            state (str, optional): State abbreviation
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        # Prepare parameters for API request
        params = {
            "api_key": self.api_key,
            "limit": self.batch_size,
            "offset": 0
        }
        
        # Add optional parameters if provided
        if county:
            params["county"] = county
        
        if state:
            params["state"] = state
        
        all_properties = []
        more_data = True
        
        # Paginate through all results
        while more_data:
            logger.debug(f"Fetching PACS data with offset {params['offset']}")
            
            # Make API request
            response = self._make_api_request("properties", params)
            
            # Process response
            if response and "properties" in response and response["properties"]:
                properties = response["properties"]
                all_properties.extend(properties)
                params["offset"] += self.batch_size
                
                logger.debug(f"Retrieved {len(properties)} properties from PACS")
            else:
                more_data = False
        
        logger.info(f"Successfully extracted {len(all_properties)} properties from PACS API")
        
        # Convert to DataFrame
        if all_properties:
            return pd.DataFrame(all_properties)
        else:
            return pd.DataFrame()
    
    def _extract_from_file(self, file_path):
        """
        Extract property data from local file.
        
        Args:
            file_path (str): Path to local file containing PACS data
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        logger.info(f"Extracting PACS data from file: {file_path}")
        
        # Determine file type from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ('.xls', '.xlsx'):
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        logger.info(f"Successfully extracted {len(df)} properties from PACS file")
        return df
    
    def _make_api_request(self, endpoint, params):
        """
        Make a request to the PACS API.
        
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
            logger.error(f"API request to PACS failed: {str(e)}")
            return None
    
    def transform_and_load(self, data, db):
        """
        Transform PACS data and load it into the database.
        
        Args:
            data (pd.DataFrame): DataFrame containing PACS property data
            db: Database connection object
            
        Returns:
            int: Number of records loaded into the database
        """
        if data.empty:
            logger.info("No PACS data to transform and load")
            return 0
            
        logger.info(f"Transforming and loading {len(data)} PACS records")
        
        try:
            # Transform data
            transformed_data = self._transform_data(data)
            
            # Load data into database
            records_loaded = self._load_data(transformed_data, db)
            
            logger.info(f"Successfully loaded {records_loaded} PACS records into database")
            return records_loaded
            
        except Exception as e:
            logger.error(f"Error transforming and loading PACS data: {str(e)}", exc_info=True)
            raise
    
    def _transform_data(self, data):
        """
        Transform PACS data into the format required for database.
        
        Args:
            data (pd.DataFrame): Raw PACS data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Map column names to standardized schema
        column_mapping = {
            "parcelId": "parcel_id",
            "apn": "apn",
            "propertyAddress": "address",
            "city": "city",
            "county": "county",
            "state": "state",
            "zipCode": "zip_code",
            "propertyType": "property_type",
            "landValue": "land_value",
            "improvementValue": "improvement_value",
            "totalValue": "total_value",
            "assessmentYear": "assessment_year",
            "bedrooms": "bedrooms",
            "bathrooms": "bathrooms",
            "buildingArea": "square_feet",
            "lotSize": "lot_size",
            "yearBuilt": "year_built",
            "lastSalePrice": "last_sale_price",
            "lastSaleDate": "last_sale_date"
        }
        
        # Rename columns based on mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add source column
        df["data_source"] = "PACS"
        
        # Add import timestamp
        df["import_date"] = datetime.now()
        
        # Convert data types
        if "land_value" in df.columns:
            df["land_value"] = pd.to_numeric(df["land_value"], errors="coerce")
        
        if "improvement_value" in df.columns:
            df["improvement_value"] = pd.to_numeric(df["improvement_value"], errors="coerce")
        
        if "total_value" in df.columns:
            df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
        
        if "square_feet" in df.columns:
            df["square_feet"] = pd.to_numeric(df["square_feet"], errors="coerce")
        
        if "lot_size" in df.columns:
            df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce")
        
        if "year_built" in df.columns:
            df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
        
        if "last_sale_price" in df.columns:
            df["last_sale_price"] = pd.to_numeric(df["last_sale_price"], errors="coerce")
        
        if "last_sale_date" in df.columns:
            df["last_sale_date"] = pd.to_datetime(df["last_sale_date"], errors="coerce")
        
        return df
    
    def _load_data(self, data, db):
        """
        Load transformed data into the database.
        
        Args:
            data (pd.DataFrame): Transformed PACS data
            db: Database connection object
            
        Returns:
            int: Number of records loaded
        """
        # Use the database object to insert the data
        return db.insert_properties(data, source="PACS")
