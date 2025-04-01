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
    
    def extract(self, start_date=None, end_date=None, csv_file_path=None):
        """
        Extract property data from MLS API or CSV file.
        
        Args:
            start_date (str, optional): Start date for data extraction (YYYY-MM-DD)
            end_date (str, optional): End date for data extraction (YYYY-MM-DD)
            csv_file_path (str, optional): Path to CSV file containing MLS data.
                                           If provided, API extraction is skipped.
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        # If CSV file path is provided, extract data from CSV file instead of API
        if csv_file_path:
            logger.info(f"Extracting data from MLS CSV file: {csv_file_path}")
            try:
                # Use the CSV reading function
                return self.read_csv_data(csv_file_path)
            except Exception as e:
                logger.error(f"Error extracting data from MLS CSV file: {str(e)}", exc_info=True)
                raise
        
        # Otherwise, extract data from the MLS API
        logger.info("Extracting data from MLS API")
        
        # Set default date range if not provided
        if not start_date:
            start_date = (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            # Check if API key is available
            if not self.api_key:
                logger.error("MLS_API_KEY is required for API extraction")
                return pd.DataFrame()
                
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
            
            logger.info(f"Successfully extracted {len(all_properties)} properties from MLS API")
            
            # Convert to DataFrame
            if all_properties:
                return pd.DataFrame(all_properties)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting data from MLS API: {str(e)}", exc_info=True)
            raise
    
    def _make_api_request(self, endpoint, params):
        """
        Make a request to the MLS API.
        
        Args:
            endpoint (str): API endpoint to call
            params (dict): Parameters for the API request
            
        Returns:
            dict: JSON response from the API or None if request failed
            
        Raises:
            ConnectionError: When network connectivity issues occur after retries
            TimeoutError: When API request takes too long to complete after retries
            ValueError: For authentication or data format issues
        """
        url = f"{self.base_url}/{endpoint}"
        max_retries = 3
        retry_count = 0
        retry_delay = 2  # Initial delay in seconds
        
        while retry_count < max_retries:
            try:
                logger.debug(f"Making API request to {url} (attempt {retry_count + 1}/{max_retries})")
                response = requests.get(url, params=params, timeout=30)
                
                # Check for specific HTTP status codes and handle accordingly
                if response.status_code == 401 or response.status_code == 403:
                    error_msg = f"Authentication failed (status code: {response.status_code}). Verify MLS_API_KEY is valid."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                elif response.status_code == 404:
                    logger.warning(f"Endpoint not found: {url}")
                    return None
                
                elif response.status_code == 429:
                    retry_count += 1
                    wait_time = retry_delay * retry_count
                    logger.warning(f"Rate limit exceeded (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                # For all other non-success status codes
                response.raise_for_status()
                
                # Parse JSON response
                try:
                    data = response.json()
                    logger.debug(f"Successfully retrieved data from {endpoint}")
                    return data
                except ValueError as json_err:
                    error_msg = f"Invalid JSON response from MLS API: {str(json_err)}"
                    logger.error(error_msg)
                    # Include response content for debugging (truncated if too long)
                    content_preview = str(response.content)[:200] + "..." if len(response.content) > 200 else str(response.content)
                    logger.debug(f"Response content preview: {content_preview}")
                    return None
                    
            except requests.exceptions.Timeout:
                retry_count += 1
                wait_time = retry_delay * retry_count
                logger.warning(f"Request timed out. Retrying in {wait_time}s ({retry_count}/{max_retries})...")
                time.sleep(wait_time)
                
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                wait_time = retry_delay * retry_count
                logger.warning(f"Connection error: {str(e)}. Retrying in {wait_time}s ({retry_count}/{max_retries})...")
                time.sleep(wait_time)
            
            except requests.exceptions.RequestException as e:
                # Log the error with traceback for debugging
                logger.error(f"API request to MLS failed: {str(e)}", exc_info=True)
                
                # For certain severe errors, don't retry
                if "SSLError" in str(e) or "ProxyError" in str(e):
                    logger.error(f"Severe connection error, not retrying: {str(e)}")
                    if "SSLError" in str(e):
                        error_msg = "SSL certificate verification failed. Check network or proxy settings."
                    else:
                        error_msg = "Proxy error occurred. Check network configuration."
                    logger.error(error_msg)
                    return None
                
                # For other errors, retry
                retry_count += 1
                wait_time = retry_delay * retry_count
                logger.warning(f"Retrying in {wait_time}s ({retry_count}/{max_retries})...")
                time.sleep(wait_time)
        
        # If we've exhausted all retries, log a critical error
        if retry_count >= max_retries:
            logger.critical(f"Failed to connect to MLS API after {max_retries} attempts. Check connectivity and API status.")
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
        
    def read_csv_data(self, file_path, date_columns=None, id_column='mls_id'):
        """
        Read and clean MLS data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing MLS data
            date_columns (list, optional): List of column names that contain dates
            id_column (str, optional): Name of the column containing unique property IDs
            
        Returns:
            pd.DataFrame: Cleaned DataFrame containing MLS property data
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If duplicate property IDs are found
        """
        logger.info(f"Reading MLS data from CSV file: {file_path}")
        
        try:
            # Step 1: Read the CSV file into a pandas DataFrame
            # Use low_memory=False to avoid mixed type inference warnings
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Successfully read {len(df)} records from CSV file")
            
            # Step 2: Check for empty DataFrame
            if df.empty:
                logger.warning("CSV file contains no data")
                return df
                
            # Step 3: Get initial column info for logging
            initial_columns = df.columns.tolist()
            logger.debug(f"Columns in CSV file: {initial_columns}")
            
            # Step 4: Remove rows with all NaN values
            original_row_count = len(df)
            df = df.dropna(how='all')
            logger.debug(f"Removed {original_row_count - len(df)} completely empty rows")
            
            # Step 5: Handle missing values by column
            # For critical columns, drop rows; for non-critical columns, fill with appropriate values
            critical_columns = [id_column, 'address', 'city', 'state', 'zip_code']
            for col in critical_columns:
                if col in df.columns:
                    # Count rows before dropping
                    pre_drop_count = len(df)
                    # Drop rows with NaN in critical columns
                    df = df.dropna(subset=[col])
                    # Log how many rows were dropped
                    logger.debug(f"Dropped {pre_drop_count - len(df)} rows with missing {col}")
            
            # Step 6: Fill NaN values in non-critical columns
            # Numeric columns get filled with 0
            numeric_columns = ['list_price', 'square_feet', 'lot_size', 'year_built', 
                              'bedrooms', 'bathrooms', 'days_on_market']
            for col in numeric_columns:
                if col in df.columns:
                    # Convert to numeric first to ensure consistent data type
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Track number of NaN values
                    na_count = df[col].isna().sum()
                    # Fill NaN values with 0
                    df[col] = df[col].fillna(0)
                    logger.debug(f"Filled {na_count} missing values in {col} with 0")
            
            # String columns get filled with empty string
            string_columns = ['property_type', 'status', 'listing_agent', 'listing_office']
            for col in string_columns:
                if col in df.columns:
                    # Track number of NaN values
                    na_count = df[col].isna().sum()
                    # Fill NaN values with empty string
                    df[col] = df[col].fillna('')
                    logger.debug(f"Filled {na_count} missing values in {col} with empty string")
            
            # Step 7: Standardize date formats
            if date_columns is None:
                # Default date columns to look for
                date_columns = ['listing_date', 'last_sale_date', 'close_date']
                
            for col in date_columns:
                if col in df.columns:
                    # Track original non-null dates
                    original_date_count = df[col].notna().sum()
                    # Convert to datetime with automatic format detection
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Check how many dates were successfully parsed
                    successful_date_count = df[col].notna().sum()
                    logger.debug(f"Converted {successful_date_count} out of {original_date_count} date values in {col} to standard format")
                    # Fill NaN dates with None for SQL compatibility
                    df[col] = df[col].fillna(pd.NaT)
            
            # Step 8: Verify that property IDs are unique
            if id_column in df.columns:
                # Check for duplicate IDs
                duplicates = df[df.duplicated(subset=[id_column], keep=False)]
                if not duplicates.empty:
                    duplicate_count = len(duplicates)
                    duplicate_ids = duplicates[id_column].unique().tolist()
                    logger.warning(f"Found {duplicate_count} rows with duplicate {id_column} values: {duplicate_ids[:5]}...")
                    
                    # Option 1: Keep the first occurrence of each duplicate ID
                    # df = df.drop_duplicates(subset=[id_column], keep='first')
                    
                    # Option 2: Raise an error
                    raise ValueError(f"Found {duplicate_count} duplicate property IDs in CSV file")
                    
                    # Option 3: Add a suffix to make IDs unique
                    # duplicate_ids = duplicates[id_column].unique()
                    # for dup_id in duplicate_ids:
                    #     dup_rows = df[df[id_column] == dup_id]
                    #     for i, idx in enumerate(dup_rows.index[1:], 1):
                    #         df.loc[idx, id_column] = f"{dup_id}_dup{i}"
            
            # Step 9: Ensure consistent data types
            # Convert string columns to string type explicitly
            for col in df.columns:
                # Skip date columns and known numeric columns
                if col not in date_columns and col not in numeric_columns:
                    # Convert to string if it's not a numeric column
                    if df[col].dtype != 'object':
                        df[col] = df[col].astype(str)
            
            # Step 10: Log cleaning results
            logger.info(f"Successfully cleaned CSV data: {len(df)} records after cleaning")
            return df
            
        except FileNotFoundError:
            logger.error(f"CSV file not found: {file_path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {str(e)}")
            raise
        except ValueError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing CSV file: {str(e)}", exc_info=True)
            raise
