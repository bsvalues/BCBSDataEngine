"""
Database management module for the property valuation system.
Provides functionality to interact with the database.
"""
import os
import logging
import json
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from db.models import Base, Property, ValidationResult

logger = logging.getLogger(__name__)

class Database:
    """
    Database management class.
    Provides methods to interact with the property database.
    """
    
    def __init__(self):
        """
        Initialize database connection.
        """
        self.database_url = os.getenv("DATABASE_URL")
        
        if not self.database_url:
            logger.warning("DATABASE_URL not found in environment variables")
            # Use a default PostgreSQL connection string
            self.database_url = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"
        
        try:
            self.engine = create_engine(
                self.database_url,
                pool_recycle=300,
                pool_pre_ping=True
            )
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established")
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}", exc_info=True)
            raise
    
    def insert_properties(self, properties_df, source):
        """
        Insert or update properties in the database.
        
        Args:
            properties_df (pd.DataFrame): DataFrame containing property data
            source (str): Data source name (MLS, NARRPR, PACS)
            
        Returns:
            int: Number of records inserted/updated
        """
        if properties_df.empty:
            return 0
            
        logger.info(f"Inserting/updating {len(properties_df)} properties from {source}")
        
        session = self.Session()
        try:
            records_processed = 0
            
            # Process in batches to avoid memory issues
            for i in range(0, len(properties_df), 100):
                batch = properties_df.iloc[i:i+100]
                
                for _, row in batch.iterrows():
                    # Create property dictionary from row
                    property_dict = row.to_dict()
                    
                    # Handle NaN values
                    for key, value in property_dict.items():
                        if pd.isna(value):
                            property_dict[key] = None
                    
                    # Add import timestamp if not present
                    if "import_date" not in property_dict or property_dict["import_date"] is None:
                        property_dict["import_date"] = datetime.now()
                    
                    # Check if property already exists
                    existing_property = None
                    
                    # Try to find by parcel_id if available
                    if "parcel_id" in property_dict and property_dict["parcel_id"]:
                        existing_property = session.query(Property).filter_by(
                            parcel_id=property_dict["parcel_id"]
                        ).first()
                    
                    # If not found and property_id is available, try that
                    if not existing_property and "property_id" in property_dict and property_dict["property_id"]:
                        existing_property = session.query(Property).filter_by(
                            property_id=property_dict["property_id"]
                        ).first()
                    
                    # If still not found, try by address
                    if (not existing_property and 
                        "address" in property_dict and property_dict["address"] and
                        "city" in property_dict and property_dict["city"] and
                        "state" in property_dict and property_dict["state"] and
                        "zip_code" in property_dict and property_dict["zip_code"]):
                        
                        existing_property = session.query(Property).filter_by(
                            address=property_dict["address"],
                            city=property_dict["city"],
                            state=property_dict["state"],
                            zip_code=property_dict["zip_code"]
                        ).first()
                    
                    # Update existing property or create new one
                    if existing_property:
                        # Keep track of original data source
                        original_source = existing_property.data_source
                        
                        # Update fields
                        for key, value in property_dict.items():
                            # Skip ID field
                            if key == "id":
                                continue
                                
                            # If current source is different from the original source,
                            # only update if the field is None or the new source has priority
                            if source != original_source:
                                current_value = getattr(existing_property, key, None)
                                if current_value is not None and not self._source_has_priority(source, original_source, key):
                                    continue
                            
                            # Set the value if the property has this attribute
                            if hasattr(existing_property, key):
                                setattr(existing_property, key, value)
                        
                        # Update the data_source field to indicate multiple sources
                        if source != original_source:
                            existing_property.data_source = f"{original_source},{source}"
                    else:
                        # Create new property
                        property_dict["data_source"] = source
                        
                        # Remove ID if present (let the database auto-generate it)
                        if "id" in property_dict:
                            del property_dict["id"]
                        
                        new_property = Property(**property_dict)
                        session.add(new_property)
                    
                    records_processed += 1
                
                # Commit after each batch
                session.commit()
                logger.debug(f"Processed batch of {len(batch)} properties")
            
            logger.info(f"Successfully processed {records_processed} properties from {source}")
            return records_processed
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting properties: {str(e)}", exc_info=True)
            raise
        finally:
            session.close()
    
    def _source_has_priority(self, new_source, existing_source, field):
        """
        Determine if the new data source has priority over the existing one for a specific field.
        
        Args:
            new_source (str): New data source
            existing_source (str): Existing data source
            field (str): Field name
            
        Returns:
            bool: True if the new source has priority, False otherwise
        """
        # Define source priority for different fields
        # Higher number means higher priority
        source_priority = {
            "MLS": {
                "list_price": 3,
                "listing_date": 3,
                "status": 3,
                "bedrooms": 2,
                "bathrooms": 2,
                "square_feet": 2,
                "property_type": 2,
                "default": 1
            },
            "NARRPR": {
                "estimated_value": 3,
                "last_sale_price": 2,
                "last_sale_date": 2,
                "year_built": 2,
                "square_feet": 2,
                "default": 1
            },
            "PACS": {
                "parcel_id": 3,
                "apn": 3,
                "land_value": 3,
                "improvement_value": 3,
                "total_value": 3,
                "assessment_year": 3,
                "year_built": 3,
                "lot_size": 3,
                "last_sale_price": 3,
                "last_sale_date": 3,
                "default": 2
            }
        }
        
        # Get priority for the new source
        new_priority = source_priority.get(new_source, {}).get(field, 
                        source_priority.get(new_source, {}).get("default", 0))
        
        # Get priority for the existing source
        existing_priority = source_priority.get(existing_source, {}).get(field, 
                          source_priority.get(existing_source, {}).get("default", 0))
        
        return new_priority >= existing_priority
    
    def get_all_properties(self, benton_county_only=True):
        """
        Get all properties from the database.
        
        Args:
            benton_county_only (bool, optional): Whether to filter results to only Benton County, WA
            
        Returns:
            pd.DataFrame: DataFrame containing all properties
        """
        logger.info("Retrieving all properties from database")
        
        try:
            # Execute query to get all properties, with optional Benton County filter
            if benton_county_only:
                query = "SELECT * FROM properties WHERE county = 'Benton' AND state = 'WA'"
                logger.info("Filtering to only Benton County, WA properties")
            else:
                query = "SELECT * FROM properties"
            
            df = pd.read_sql(query, self.engine)
            
            logger.info(f"Retrieved {len(df)} properties from database")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving properties: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def get_properties_by_criteria(self, criteria=None, benton_county_only=True):
        """
        Get properties from the database based on criteria.
        
        Args:
            criteria (dict, optional): Dictionary of criteria to filter properties
            benton_county_only (bool, optional): Whether to filter results to only Benton County, WA
            
        Returns:
            pd.DataFrame: DataFrame containing filtered properties
        """
        if not criteria:
            return self.get_all_properties(benton_county_only=benton_county_only)
        
        logger.info(f"Retrieving properties with criteria: {criteria}")
        
        try:
            # Build the query based on criteria
            query = "SELECT * FROM properties WHERE "
            conditions = []
            params = {}
            
            # Add Benton County, WA filter if specified
            if benton_county_only:
                conditions.append("county = :county AND state = :state")
                params["county"] = "Benton"
                params["state"] = "WA"
                logger.info("Filtering to only Benton County, WA properties")
            
            for key, value in criteria.items():
                if key == "min_price":
                    conditions.append("list_price >= :min_price")
                    params["min_price"] = value
                elif key == "max_price":
                    conditions.append("list_price <= :max_price")
                    params["max_price"] = value
                elif key == "min_square_feet":
                    conditions.append("square_feet >= :min_square_feet")
                    params["min_square_feet"] = value
                elif key == "max_square_feet":
                    conditions.append("square_feet <= :max_square_feet")
                    params["max_square_feet"] = value
                elif key == "property_type":
                    conditions.append("property_type = :property_type")
                    params["property_type"] = value
                elif key == "city":
                    conditions.append("city = :city")
                    params["city"] = value
                elif key == "state":
                    conditions.append("state = :state")
                    params["state"] = value
                elif key == "zip_code":
                    conditions.append("zip_code = :zip_code")
                    params["zip_code"] = value
                elif key == "min_year_built":
                    conditions.append("year_built >= :min_year_built")
                    params["min_year_built"] = value
                elif key == "max_year_built":
                    conditions.append("year_built <= :max_year_built")
                    params["max_year_built"] = value
                elif key == "min_bedrooms":
                    conditions.append("bedrooms >= :min_bedrooms")
                    params["min_bedrooms"] = value
                elif key == "max_bedrooms":
                    conditions.append("bedrooms <= :max_bedrooms")
                    params["max_bedrooms"] = value
                elif key == "min_bathrooms":
                    conditions.append("bathrooms >= :min_bathrooms")
                    params["min_bathrooms"] = value
                elif key == "max_bathrooms":
                    conditions.append("bathrooms <= :max_bathrooms")
                    params["max_bathrooms"] = value
                elif key == "data_source":
                    conditions.append("data_source = :data_source")
                    params["data_source"] = value
            
            if not conditions:
                return self.get_all_properties()
            
            query += " AND ".join(conditions)
            
            # Execute query with parameters
            df = pd.read_sql(text(query), self.engine, params=params)
            
            logger.info(f"Retrieved {len(df)} properties matching criteria")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving properties by criteria: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def store_validation_results(self, results):
        """
        Store validation results in the database.
        
        Args:
            results (dict): Validation results
        """
        logger.info("Storing validation results")
        
        session = self.Session()
        try:
            # Convert results to JSON string
            results_json = json.dumps(results)
            
            # Create new validation result
            validation_result = ValidationResult(
                timestamp=datetime.now(),
                status=results["status"],
                results=results_json
            )
            
            session.add(validation_result)
            session.commit()
            
            logger.info("Validation results stored successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing validation results: {str(e)}", exc_info=True)
        finally:
            session.close()
    
    def get_validation_results(self, limit=10):
        """
        Get recent validation results from the database.
        
        Args:
            limit (int, optional): Maximum number of results to retrieve
            
        Returns:
            list: List of validation results
        """
        logger.info(f"Retrieving up to {limit} recent validation results")
        
        session = self.Session()
        try:
            # Query for recent validation results
            results = session.query(ValidationResult).order_by(
                ValidationResult.timestamp.desc()
            ).limit(limit).all()
            
            # Convert to list of dictionaries
            validation_results = []
            for result in results:
                validation_results.append({
                    "id": result.id,
                    "timestamp": result.timestamp.isoformat(),
                    "status": result.status,
                    "results": json.loads(result.results)
                })
            
            logger.info(f"Retrieved {len(validation_results)} validation results")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error retrieving validation results: {str(e)}", exc_info=True)
            return []
        finally:
            session.close()
    
    def batch_insert_properties(self, properties_df, batch_size=1000):
        """
        Efficiently insert a DataFrame of properties into the database using batch insertion.
        
        This method uses SQLAlchemy's Core functionality for bulk insertion, which is more
        efficient than ORM-based insertion for large datasets.
        
        Args:
            properties_df (pd.DataFrame): DataFrame containing property data
            batch_size (int, optional): Number of records to insert in each batch
            
        Returns:
            int: Number of records inserted
            
        Raises:
            SQLAlchemyError: If there's an error with the database connection or insertion
            ValueError: If the DataFrame is empty or doesn't match the expected schema
        """
        if properties_df.empty:
            logger.warning("Empty DataFrame provided, no records to insert")
            return 0
            
        logger.info(f"Batch inserting {len(properties_df)} property records")
        
        try:
            # Ensure the engine is connected
            if not hasattr(self, 'engine') or self.engine is None:
                raise ValueError("Database connection not established")
                
            # Make a copy of the DataFrame to avoid modifying the original
            df = properties_df.copy()
            
            # Handle NaN values - convert to None for SQL compatibility
            df = df.where(pd.notnull(df), None)
            
            # Add import timestamp if not present
            if 'import_date' not in df.columns or df['import_date'].isnull().all():
                df['import_date'] = datetime.now()
                
            # Drop the 'id' column if it exists (let the database auto-generate it)
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
                
            # Get the Property table columns for validation
            # This helps catch mismatched columns before attempting insertion
            property_columns = [c.name for c in Property.__table__.columns if c.name != 'id']
            
            # Filter the DataFrame to include only valid columns
            valid_columns = [col for col in df.columns if col in property_columns]
            if not valid_columns:
                raise ValueError(f"No valid Property model columns found in DataFrame. Expected columns: {property_columns}")
                
            df = df[valid_columns]
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient='records')
            total_records = len(records)
            records_inserted = 0
            
            # Batch insert records
            # This is much more efficient than inserting one by one
            for i in range(0, total_records, batch_size):
                batch = records[i:i + batch_size]
                if batch:
                    try:
                        # Use SQLAlchemy Core for batch insertion
                        with self.engine.begin() as conn:
                            # The execute() method automatically creates a transaction
                            # that will be committed if successful or rolled back on error
                            conn.execute(Property.__table__.insert(), batch)
                            
                        records_inserted += len(batch)
                        
                        # CREATE INDEX recommendation:
                        # For best performance with large datasets, consider creating indexes:
                        # CREATE INDEX idx_properties_address ON properties(address, city, state, zip_code);
                        # CREATE INDEX idx_properties_location ON properties(city, state, zip_code);
                        # CREATE INDEX idx_properties_parcel ON properties(parcel_id);
                        # CREATE INDEX idx_properties_property ON properties(property_id);
                        logger.debug(f"Inserted batch of {len(batch)} records (total: {records_inserted}/{total_records})")
                    except SQLAlchemyError as e:
                        # Log the specific error but continue with the next batch
                        logger.error(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
                        raise
            
            logger.info(f"Successfully inserted {records_inserted} property records")
            return records_inserted
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during batch insertion: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch insertion: {str(e)}", exc_info=True)
            raise
            
    def create_properties_indexes(self):
        """
        Create indexes on the properties table to improve query performance.
        
        This method creates indexes on commonly queried fields in the properties table.
        It's recommended to call this after large batch insertions to improve query performance.
        
        Note: Creating indexes can be a time-consuming operation on large tables, but
        the performance benefits for queries are substantial.
        
        Returns:
            bool: True if indexes were created successfully, False otherwise
        """
        logger.info("Creating indexes on properties table")
        
        # SQL statements to create indexes
        index_statements = [
            # Composite index for full address searches
            "CREATE INDEX IF NOT EXISTS idx_properties_address ON properties(address, city, state, zip_code)",
            
            # Index for location-based searches
            "CREATE INDEX IF NOT EXISTS idx_properties_location ON properties(city, state, zip_code)",
            
            # Indexes for identifier fields
            "CREATE INDEX IF NOT EXISTS idx_properties_parcel_id ON properties(parcel_id)",
            "CREATE INDEX IF NOT EXISTS idx_properties_property_id ON properties(property_id)",
            "CREATE INDEX IF NOT EXISTS idx_properties_mls_id ON properties(mls_id)",
            
            # Indexes for common filter criteria
            "CREATE INDEX IF NOT EXISTS idx_properties_price ON properties(list_price)",
            "CREATE INDEX IF NOT EXISTS idx_properties_sqft ON properties(square_feet)",
            "CREATE INDEX IF NOT EXISTS idx_properties_bedrooms ON properties(bedrooms)",
            "CREATE INDEX IF NOT EXISTS idx_properties_property_type ON properties(property_type)",
            
            # Index for data source
            "CREATE INDEX IF NOT EXISTS idx_properties_data_source ON properties(data_source)"
        ]
        
        try:
            with self.engine.begin() as conn:
                for statement in index_statements:
                    try:
                        conn.execute(text(statement))
                        logger.debug(f"Created index: {statement}")
                    except SQLAlchemyError as e:
                        logger.warning(f"Error creating index: {statement}. Error: {str(e)}")
                
            logger.info("Indexes created successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating indexes: {str(e)}", exc_info=True)
            return False
        
    def close(self):
        """
        Close the database connection.
        """
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connection closed")
