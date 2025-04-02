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
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"PACS CSV file not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file extension and read accordingly
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.csv':
                # Read CSV file with appropriate options
                # - Parse dates for date columns
                # - Handle different date formats commonly found in PACS data
                # - Specify NA values that might be in the legacy data
                # Read header first to check available columns
                header = pd.read_csv(file_path, nrows=0)
                # Determine which date columns are present in the file
                date_columns = [col for col in ['AssessmentDate', 'SaleDate', 'RecordDate'] if col in header.columns]
                
                # Read full file with appropriate date parsing
                data = pd.read_csv(
                    file_path,
                    parse_dates=date_columns,
                    date_format='mixed',  # Try to infer date format
                    na_values=['NA', 'N/A', '#N/A', 'NULL', ''],
                    low_memory=False  # Avoid dtype warnings for mixed columns
                )
            elif ext.lower() in ['.xlsx', '.xls']:
                # Read header first to check available columns
                header = pd.read_excel(file_path, nrows=0)
                # Determine which date columns are present in the file
                date_columns = [col for col in ['AssessmentDate', 'SaleDate', 'RecordDate'] if col in header.columns]
                
                # Handle Excel files
                data = pd.read_excel(
                    file_path,
                    parse_dates=date_columns,
                    na_values=['NA', 'N/A', '#N/A', 'NULL', '']
                )
            else:
                logger.error(f"Unsupported file format: {ext}")
                raise ValueError(f"Unsupported file format: {ext}. Expected .csv, .xlsx, or .xls")
            
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
        """
        # Check if data is empty
        if data.empty:
            logger.warning("No data to load into database")
            return 0
        
        # Insert properties into database
        try:
            records_loaded = db.insert_properties(data, source='PACS')
            logger.info(f"Loaded {records_loaded} PACS records into database")
            return records_loaded
        except Exception as e:
            logger.error(f"Error loading PACS data into database: {str(e)}")
            raise
    
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