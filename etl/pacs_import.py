"""
PACS (Property Appraisal & Classification System) data import module.
Handles extraction of property data from PACS CSV files, transformation, and loading.
"""
import os
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class PACSImporter:
    """
    Importer for extracting, transforming, and loading data from PACS (Property Appraisal & Classification System) files.
    PACS data typically contains property assessment information from county/municipal tax authorities.
    """
    
    def __init__(self, batch_size=100):
        """
        Initialize the PACS importer.
        
        Args:
            batch_size (int): Number of records to process in each batch
        """
        self.batch_size = batch_size
        self.api_key = os.environ.get('PACS_API_KEY')
        
        if not self.api_key:
            logger.warning("PACS_API_KEY not found in environment variables")
    
    def extract(self, file_path=None):
        """
        Extract property data from PACS CSV files or API.
        
        Args:
            file_path (str, optional): Path to CSV file containing PACS data.
                                       If provided, API extraction is skipped.
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        if file_path:
            return self._extract_from_csv(file_path)
        else:
            return self._extract_from_api()
    
    def _extract_from_csv(self, file_path):
        """
        Extract property data from a PACS CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing PACS data
            
        Returns:
            pd.DataFrame: DataFrame containing PACS property data
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
        """
        logger.info(f"Extracting PACS data from CSV file: {file_path}")
        
        try:
            # Check if file exists with detailed error messages
            if not os.path.exists(file_path):
                error_msg = f"PACS file not found: {file_path}"
                logger.error(error_msg)
                # Add directory listing for debugging
                try:
                    parent_dir = os.path.dirname(file_path)
                    if os.path.exists(parent_dir):
                        files_in_dir = os.listdir(parent_dir)
                        logger.debug(f"Files in directory {parent_dir}: {files_in_dir}")
                    else:
                        logger.debug(f"Parent directory {parent_dir} does not exist")
                except Exception as list_error:
                    logger.debug(f"Could not list directory contents: {str(list_error)}")
                    
                raise FileNotFoundError(error_msg)
            
            # Check if file is readable and not empty
            if os.path.getsize(file_path) == 0:
                error_msg = f"PACS file is empty: {file_path}"
                logger.error(error_msg)
                return pd.DataFrame()  # Return empty DataFrame for empty file
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                error_msg = f"PACS file is not readable: {file_path}"
                logger.error(error_msg)
                raise PermissionError(error_msg)
            
            # Determine file extension and read accordingly
            _, ext = os.path.splitext(file_path)
            logger.info(f"Processing PACS file with extension: {ext}")
            
            if ext.lower() == '.csv':
                # First try to read just a few rows to validate file format
                try:
                    # Read header first to check available columns
                    header = pd.read_csv(file_path, nrows=0)
                    logger.debug(f"CSV columns found: {header.columns.tolist()}")
                    
                    # Count total lines for progress reporting
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    logger.info(f"CSV file contains approximately {line_count} lines")
                    
                    # Determine which date columns are present in the file
                    date_columns = [col for col in ['AssessmentDate', 'SaleDate', 'RecordDate'] if col in header.columns]
                    logger.debug(f"Date columns detected: {date_columns}")
                    
                    # Read full file with appropriate date parsing and detailed error handling
                    data = pd.read_csv(
                        file_path,
                        parse_dates=date_columns,
                        date_format='mixed',  # Try to infer date format
                        na_values=['NA', 'N/A', '#N/A', 'NULL', ''],
                        low_memory=False,  # Avoid dtype warnings for mixed columns
                        on_bad_lines='warn'  # Log but skip bad lines instead of failing
                    )
                    logger.info(f"Successfully read {len(data)} records from CSV file")
                    
                except pd.errors.ParserError as pe:
                    logger.error(f"CSV parsing error: {str(pe)}", exc_info=True)
                    # Try more robust parsing with error recovery
                    logger.info("Attempting to recover from CSV parsing error...")
                    try:
                        # Attempt with error_bad_lines=False to skip problematic rows
                        data = pd.read_csv(
                            file_path,
                            parse_dates=date_columns if 'date_columns' in locals() else None,
                            date_format='mixed',
                            na_values=['NA', 'N/A', '#N/A', 'NULL', ''],
                            low_memory=False,
                            on_bad_lines='skip'  # Skip bad lines entirely
                        )
                        logger.warning(f"Recovered by skipping bad lines. Read {len(data)} records.")
                    except Exception as recovery_error:
                        logger.critical(f"Recovery attempt also failed: {str(recovery_error)}", exc_info=True)
                        raise ValueError(f"Could not parse CSV file even with recovery options: {str(pe)}") from pe
                    
                except Exception as e:
                    logger.error(f"Unexpected error reading CSV header: {str(e)}", exc_info=True)
                    raise
                    
            elif ext.lower() in ['.xlsx', '.xls']:
                try:
                    # Read header first to check available columns
                    header = pd.read_excel(file_path, nrows=0)
                    logger.debug(f"Excel columns found: {header.columns.tolist()}")
                    
                    # Determine which date columns are present in the file
                    date_columns = [col for col in ['AssessmentDate', 'SaleDate', 'RecordDate'] if col in header.columns]
                    logger.debug(f"Date columns detected: {date_columns}")
                    
                    # Handle Excel files with error monitoring
                    data = pd.read_excel(
                        file_path,
                        parse_dates=date_columns,
                        na_values=['NA', 'N/A', '#N/A', 'NULL', '']
                    )
                    logger.info(f"Successfully read {len(data)} records from Excel file")
                    
                except ImportError as ie:
                    # This would occur if openpyxl or xlrd is not installed
                    error_msg = f"Excel support libraries not installed: {str(ie)}"
                    logger.error(error_msg, exc_info=True)
                    logger.info("Attempting to install required dependencies...")
                    raise ImportError(f"{error_msg}. Please install openpyxl or xlrd package.")
                
                except Exception as e:
                    logger.error(f"Error reading Excel file: {str(e)}", exc_info=True)
                    raise
            else:
                error_msg = f"Unsupported file extension: {ext}. Only .csv, .xlsx, and .xls are supported."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log data dimensions and preview
            logger.info(f"Loaded {len(data)} records from {file_path}")
            logger.debug(f"PACS data preview: {data.head(2)}")
            logger.debug(f"PACS data columns: {data.columns.tolist()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting PACS data from CSV: {str(e)}")
            raise
    
    def _extract_from_api(self):
        """
        Extract property data from PACS API (if available).
        
        Returns:
            pd.DataFrame: DataFrame containing PACS property data
        """
        logger.info("PACS API extraction not implemented yet")
        logger.info("Use file-based extraction by providing a file_path")
        
        # Return empty DataFrame
        return pd.DataFrame()
    
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
            logger.warning("No PACS data to transform and load")
            return 0
        
        logger.info(f"Transforming {len(data)} PACS records")
        
        # Transform the data
        transformed_data = self._transform_data(data)
        
        # Load the data into the database
        records_loaded = self._load_data(transformed_data, db)
        
        return records_loaded
    
    def _transform_data(self, data):
        """
        Transform PACS data into the format required for database.
        This method handles the mapping from PACS schema to the unified property schema.
        
        Args:
            data (pd.DataFrame): Raw PACS data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        logger.info("Transforming PACS data to match unified property schema")
        
        # Make a copy to avoid modifying the original
        transformed = data.copy()
        
        # Step 1: Rename columns from PACS schema to our unified schema
        # Common PACS column names are mapped to our schema's names
        column_mapping = {
            # Identification fields
            'ParcelID': 'parcel_id',
            'APN': 'apn',
            'AssessorParcelNum': 'apn',
            'PropertyID': 'property_id',
            
            # Location fields
            'SitusAddress': 'address',
            'PropertyAddress': 'address',
            'SitusCity': 'city',
            'City': 'city',
            'SitusState': 'state',
            'State': 'state',
            'SitusZip': 'zip_code',
            'ZipCode': 'zip_code',
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'County': 'county',
            
            # Property characteristics
            'PropertyType': 'property_type',
            'LandUseCode': 'property_type',
            'Bedrooms': 'bedrooms',
            'Bathrooms': 'bathrooms',
            'TotalRooms': 'total_rooms',
            'LivingSqFt': 'square_feet',
            'BuildingSqFt': 'square_feet',
            'LotSize': 'lot_size',
            'LotSizeAcres': 'lot_size',
            'YearBuilt': 'year_built',
            'Stories': 'stories',
            'Basement': 'basement',
            'Garage': 'garage',
            'GarageSpaces': 'garage_spaces',
            'HasPool': 'pool',
            'View': 'view',
            'ConstructionType': 'construction_type',
            'RoofType': 'roof_type',
            'Foundation': 'foundation_type',
            
            # Valuation fields
            'ListPrice': 'list_price',
            'MarketValue': 'estimated_value',
            'LastSalePrice': 'last_sale_price',
            'LastSaleAmount': 'last_sale_price',
            'SaleDate': 'last_sale_date',
            'LastSaleDate': 'last_sale_date',
            'LandValue': 'land_value',
            'ImprovementValue': 'improvement_value',
            'TotalAssessedValue': 'total_value',
            'AssessmentYear': 'assessment_year',
            'AssessmentDate': 'assessment_date',
            
            # Status fields
            'ListingDate': 'listing_date',
            'PropertyStatus': 'status',
            'DaysOnMarket': 'days_on_market',
            'ListingAgent': 'listing_agent',
            'ListingOffice': 'listing_office'
        }
        
        # Apply column renaming for columns that exist in the data
        cols_to_rename = {old: new for old, new in column_mapping.items() if old in transformed.columns}
        transformed.rename(columns=cols_to_rename, inplace=True)
        
        # Step 2: Add data source and import date
        transformed['data_source'] = 'PACS'
        transformed['import_date'] = datetime.now()
        
        # Step 3: Handle special transformations for PACS-specific data
        
        # Convert property type codes to descriptive strings
        if 'property_type' in transformed.columns:
            # PACS often uses numeric codes for property types
            # Map common PACS property type codes to descriptive text
            pacs_property_type_map = {
                'R': 'Residential',
                'C': 'Commercial',
                'A': 'Agricultural',
                'I': 'Industrial',
                'E': 'Exempt',
                'M': 'Multi-Family',
                'V': 'Vacant',
                '100': 'Single Family Residential',
                '101': 'Single Family Residential',
                '200': 'Multi-Family',
                '300': 'Commercial',
                '400': 'Industrial',
                '500': 'Agricultural',
                '600': 'Vacant Land',
                '700': 'Exempt'
            }
            
            # Apply mapping where applicable
            transformed['property_type'] = transformed['property_type'].map(lambda x: pacs_property_type_map.get(str(x), x))
        
        # Step 4: Convert units where necessary
        
        # Convert lot size from acres to square feet if in acres
        if 'lot_size' in transformed.columns and 'LotSizeAcres' in cols_to_rename:
            # Multiply acres by 43,560 to get square feet
            transformed['lot_size'] = transformed['lot_size'] * 43560
        
        # Step 5: Standardize boolean fields
        
        # Convert pool indicator to Yes/No
        if 'pool' in transformed.columns:
            # Map various pool indicators to Yes/No format
            pool_map = {
                'Y': 'Yes',
                'N': 'No',
                'YES': 'Yes',
                'NO': 'No',
                'Y/YES': 'Yes',
                'N/NO': 'No',
                True: 'Yes',
                False: 'No',
                1: 'Yes',
                0: 'No'
            }
            transformed['pool'] = transformed['pool'].map(lambda x: pool_map.get(x, 'No' if pd.isna(x) else str(x)))
        
        # Similar mapping for basement
        if 'basement' in transformed.columns:
            basement_map = {
                'Y': 'Yes',
                'N': 'No',
                'YES': 'Yes',
                'NO': 'No',
                'FULL': 'Full',
                'PARTIAL': 'Partial',
                'FINISHED': 'Finished',
                'UNFINISHED': 'Unfinished',
                True: 'Yes',
                False: 'No',
                1: 'Yes',
                0: 'No'
            }
            transformed['basement'] = transformed['basement'].map(lambda x: basement_map.get(x, 'No' if pd.isna(x) else str(x)))
        
        # Step 6: Extract county from address if not available
        if 'county' not in transformed.columns and 'address' in transformed.columns:
            # Attempt to extract county from address in some PACS formats
            logger.info("County field not found, attempting to extract from address")
            # This would be a more complex regex pattern in a real implementation
        
        # Step 7: Handle data type conversions
        
        # Ensure numeric fields are numeric
        numeric_fields = [
            'bedrooms', 'bathrooms', 'total_rooms', 'square_feet', 'lot_size',
            'year_built', 'stories', 'garage_spaces', 'list_price', 'estimated_value',
            'last_sale_price', 'land_value', 'improvement_value', 'total_value',
            'assessment_year', 'days_on_market', 'latitude', 'longitude'
        ]
        
        for field in numeric_fields:
            if field in transformed.columns:
                # Convert to numeric, errors='coerce' will set invalid parsing to NaN
                transformed[field] = pd.to_numeric(transformed[field], errors='coerce')
                
        # Ensure string fields are strings
        string_fields = [
            'parcel_id', 'property_id', 'apn', 'address', 'city', 'county', 
            'state', 'zip_code', 'basement', 'garage', 'pool', 'view',
            'construction_type', 'roof_type', 'foundation_type', 'status',
            'listing_agent', 'listing_office', 'data_source'
        ]
        
        for field in string_fields:
            if field in transformed.columns:
                # Convert to string
                transformed[field] = transformed[field].astype(str)
        
        # Ensure date fields are datetime objects
        date_fields = [
            'last_sale_date', 'listing_date', 'assessment_date', 'import_date'
        ]
        
        for field in date_fields:
            if field in transformed.columns and field != 'import_date':  # import_date is already datetime
                # Convert to datetime, errors='coerce' will set invalid parsing to NaT
                transformed[field] = pd.to_datetime(transformed[field], errors='coerce')
        
        # Step 8: Fill missing values with appropriate defaults
        
        # Fill missing values for key fields with appropriate defaults
        string_defaults = {
            'property_type': 'Unknown',
            'status': 'Unknown'
        }
        
        # Apply string defaults
        for field, default_value in string_defaults.items():
            if field in transformed.columns:
                transformed[field] = transformed[field].fillna(default_value)
        
        # Numeric fields can remain as NaN (pandas will handle them properly)
        # We don't need to explicitly set None for numeric fields
        
        # Step 9: Log transformed data stats
        logger.info(f"Transformation complete: {len(transformed)} records")
        logger.debug(f"Transformed PACS data columns: {transformed.columns.tolist()}")
        
        # Remove any extra columns not in our schema
        standard_columns = [
            'id', 'mls_id', 'listing_id', 'property_id', 'parcel_id', 'apn',
            'address', 'city', 'county', 'state', 'zip_code', 'latitude', 'longitude',
            'property_type', 'bedrooms', 'bathrooms', 'total_rooms', 'square_feet',
            'lot_size', 'year_built', 'stories', 'basement', 'garage', 'garage_spaces',
            'pool', 'view', 'construction_type', 'roof_type', 'foundation_type',
            'list_price', 'estimated_value', 'last_sale_price', 'last_sale_date',
            'land_value', 'improvement_value', 'total_value', 'assessment_year',
            'listing_date', 'status', 'days_on_market', 'listing_agent', 'listing_office',
            'data_source', 'import_date'
        ]
        
        # Only keep columns that are in our standard schema (if they exist in transformed)
        keep_columns = [col for col in standard_columns if col in transformed.columns]
        final_df = transformed[keep_columns]
        
        return final_df
    
    def _load_data(self, data, db):
        """
        Load transformed data into the database.
        
        Args:
            data (pd.DataFrame): Transformed PACS data
            db: Database connection object
            
        Returns:
            int: Number of records loaded
            
        Raises:
            ValueError: If data validation fails
            IOError: If database connection fails
            RuntimeError: For other database errors
        """
        # Check if data is empty
        if data.empty:
            logger.warning("No data to load into database")
            return 0
        
        # Verify database connection before attempting to load data
        if db is None:
            error_msg = "Database connection is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Log database loading attempt with data information
        logger.info(f"Attempting to load {len(data)} PACS records into database")
        logger.debug(f"Data columns to be loaded: {data.columns.tolist()}")
        
        # Insert properties into database with staged loading and error handling
        total_records_loaded = 0
        batch_size = min(1000, len(data))  # Load in batches of 1000 or less
        
        try:
            # Validate data before loading
            if 'address' not in data.columns or 'property_type' not in data.columns:
                error_msg = "Data is missing required columns: address and property_type are required"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Process data in batches for better error handling and performance
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size].copy()
                
                try:
                    # Attempt to load this batch
                    logger.debug(f"Loading batch {i//batch_size + 1} of {(len(data) + batch_size - 1)//batch_size}")
                    
                    # Insert into database
                    batch_records_loaded = db.insert_properties(batch, source='PACS')
                    total_records_loaded += batch_records_loaded
                    
                    logger.debug(f"Batch {i//batch_size + 1} loaded: {batch_records_loaded} records")
                    
                except Exception as batch_error:
                    # Log detailed error for this batch
                    error_msg = f"Error loading batch {i//batch_size + 1}: {str(batch_error)}"
                    logger.error(error_msg, exc_info=True)
                    
                    # Try to identify problematic records in this batch
                    try:
                        # Check for null values in key columns
                        null_count = batch.isnull().sum()
                        if null_count.any():
                            logger.warning(f"Batch contains null values: {null_count[null_count > 0].to_dict()}")
                        
                        # Check for duplicate IDs if we have ID columns
                        for id_col in ['property_id', 'parcel_id', 'apn']:
                            if id_col in batch.columns and not batch[id_col].isna().all():
                                duplicates = batch[batch.duplicated(subset=[id_col], keep=False)]
                                if not duplicates.empty:
                                    dup_ids = duplicates[id_col].unique().tolist()
                                    logger.warning(f"Batch contains duplicate {id_col}s: {dup_ids[:5]}...")
                    except Exception as analysis_error:
                        logger.debug(f"Error analyzing failed batch: {str(analysis_error)}")
                    
                    # Continue with next batch rather than failing all
                    logger.info("Continuing with next batch...")
                    continue
            
            logger.info(f"Successfully loaded {total_records_loaded} PACS records into database")
            if total_records_loaded < len(data):
                logger.warning(f"Not all records were loaded: {len(data) - total_records_loaded} records failed")
                
            return total_records_loaded
            
        except ValueError as ve:
            # Data validation errors
            logger.error(f"Data validation error: {str(ve)}")
            raise
            
        except Exception as e:
            # Other database errors
            error_type = type(e).__name__
            error_msg = f"Error loading PACS data into database: {error_type}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Add more context about the data for debugging
            try:
                logger.debug(f"Data types being loaded: {data.dtypes}")
                sample_records = data.head(2).to_dict(orient='records')
                logger.debug(f"Sample records: {sample_records}")
            except Exception as debug_error:
                logger.debug(f"Could not generate debug info: {str(debug_error)}")
                
            # Re-raise with more context
            raise RuntimeError(f"Database loading failed: {error_msg}") from e
    
    def read_sample_file(self, file_path, n=5):
        """
        Read the first n rows of a PACS file to preview the data.
        Useful for debugging and exploring PACS file schema.
        
        Args:
            file_path (str): Path to the PACS file
            n (int): Number of rows to read
            
        Returns:
            pd.DataFrame: DataFrame containing sample rows
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path, nrows=n)
        elif ext.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, nrows=n)
        else:
            raise ValueError(f"Unsupported file format: {ext}")