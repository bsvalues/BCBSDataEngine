"""
Data validation module for real estate property data.
Provides functionality to validate data quality and consistency.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validator for real estate property data.
    Provides methods to check data quality and consistency.
    """
    
    def __init__(self, db):
        """
        Initialize the data validator.
        
        Args:
            db: Database connection object
        """
        self.db = db
    
    def validate_all(self):
        """
        Run all validation checks on the property data.
        
        Returns:
            dict: Dictionary containing validation results
        """
        logger.info("Running comprehensive data validation")
        
        # Get all properties from the database
        properties = self.db.get_all_properties()
        
        if properties.empty:
            logger.warning("No properties found in database for validation")
            return {"status": "no_data", "validations": {}}
        
        # Run all validation checks
        validations = {}
        
        # Data completeness check
        validations["completeness"] = self.validate_completeness(properties)
        
        # Data type check
        validations["data_types"] = self.validate_data_types(properties)
        
        # Range check for numeric values
        validations["numeric_ranges"] = self.validate_numeric_ranges(properties)
        
        # Date validity check
        validations["date_validity"] = self.validate_dates(properties)
        
        # Duplicate check
        validations["duplicates"] = self.validate_duplicates(properties)
        
        # Cross-source consistency check
        validations["cross_source"] = self.validate_cross_source_consistency(properties)
        
        # Calculate overall validation status
        overall_status = "passed"
        for validation_name, validation_result in validations.items():
            if validation_result["status"] == "failed":
                overall_status = "failed"
                break
        
        return {
            "status": overall_status,
            "validations": validations,
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_completeness(self, data):
        """
        Check for missing values in critical fields.
        
        Args:
            data (pd.DataFrame): Property data
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating data completeness")
        
        # Define critical fields
        critical_fields = [
            "address", "city", "state", "zip_code", 
            "property_type", "square_feet", "year_built"
        ]
        
        # Calculate percentage of missing values for each critical field
        missing_percentages = {}
        for field in critical_fields:
            if field in data.columns:
                missing_count = data[field].isna().sum()
                missing_percentage = (missing_count / len(data)) * 100
                missing_percentages[field] = round(missing_percentage, 2)
            else:
                missing_percentages[field] = 100  # Field is completely missing
        
        # Define threshold for failure
        threshold = 10  # 10% missing is considered a failure
        
        failed_fields = {k: v for k, v in missing_percentages.items() if v > threshold}
        
        if failed_fields:
            return {
                "status": "failed",
                "message": f"Critical fields exceed missing value threshold of {threshold}%",
                "failed_fields": failed_fields,
                "all_fields": missing_percentages
            }
        else:
            return {
                "status": "passed",
                "message": "All critical fields meet completeness requirements",
                "all_fields": missing_percentages
            }
    
    def validate_data_types(self, data):
        """
        Check if data types are correct for all fields.
        
        Args:
            data (pd.DataFrame): Property data
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating data types")
        
        # Define expected data types
        expected_types = {
            "square_feet": "numeric",
            "lot_size": "numeric",
            "year_built": "numeric",
            "bedrooms": "numeric",
            "bathrooms": "numeric",
            "list_price": "numeric",
            "last_sale_price": "numeric",
            "estimated_value": "numeric",
            "land_value": "numeric",
            "improvement_value": "numeric",
            "total_value": "numeric",
            "last_sale_date": "date",
            "listing_date": "date",
            "import_date": "date"
        }
        
        type_errors = {}
        
        for field, expected_type in expected_types.items():
            if field not in data.columns:
                continue
                
            if expected_type == "numeric":
                # Check if numeric fields are actually numeric
                non_numeric = ~pd.to_numeric(data[field], errors='coerce').notna() & ~data[field].isna()
                error_count = non_numeric.sum()
                
                if error_count > 0:
                    type_errors[field] = {
                        "expected": expected_type,
                        "error_count": int(error_count),
                        "error_percentage": round((error_count / len(data)) * 100, 2)
                    }
                    
            elif expected_type == "date":
                # Check if date fields are actually dates
                non_date = ~pd.to_datetime(data[field], errors='coerce').notna() & ~data[field].isna()
                error_count = non_date.sum()
                
                if error_count > 0:
                    type_errors[field] = {
                        "expected": expected_type,
                        "error_count": int(error_count),
                        "error_percentage": round((error_count / len(data)) * 100, 2)
                    }
        
        if type_errors:
            return {
                "status": "failed",
                "message": "Data type validation failed for some fields",
                "errors": type_errors
            }
        else:
            return {
                "status": "passed",
                "message": "All fields have correct data types"
            }
    
    def validate_numeric_ranges(self, data):
        """
        Check if numeric values are within reasonable ranges.
        
        Args:
            data (pd.DataFrame): Property data
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating numeric ranges")
        
        # Define reasonable ranges for numeric fields
        ranges = {
            "square_feet": (100, 20000),  # Square feet between 100 and 20,000
            "lot_size": (0.01, 100),      # Lot size between 0.01 and 100 acres
            "year_built": (1800, datetime.now().year),  # Year built between 1800 and current year
            "bedrooms": (0, 20),          # Bedrooms between 0 and 20
            "bathrooms": (0, 15),         # Bathrooms between 0 and 15
            "list_price": (1000, 100000000),  # List price between $1,000 and $100,000,000
            "last_sale_price": (1000, 100000000)  # Sale price between $1,000 and $100,000,000
        }
        
        range_errors = {}
        
        for field, (min_val, max_val) in ranges.items():
            if field not in data.columns:
                continue
                
            # Convert to numeric, coerce errors to NaN
            numeric_values = pd.to_numeric(data[field], errors='coerce')
            
            # Count values outside the range
            out_of_range = ((numeric_values < min_val) | (numeric_values > max_val)) & numeric_values.notna()
            error_count = out_of_range.sum()
            
            if error_count > 0:
                range_errors[field] = {
                    "min": min_val,
                    "max": max_val,
                    "error_count": int(error_count),
                    "error_percentage": round((error_count / len(data)) * 100, 2)
                }
        
        if range_errors:
            return {
                "status": "failed",
                "message": "Numeric range validation failed for some fields",
                "errors": range_errors
            }
        else:
            return {
                "status": "passed",
                "message": "All numeric fields are within reasonable ranges"
            }
    
    def validate_dates(self, data):
        """
        Check if dates are valid and in the correct range.
        
        Args:
            data (pd.DataFrame): Property data
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating dates")
        
        # Define date fields and their valid ranges
        date_fields = {
            "last_sale_date": datetime(1900, 1, 1),  # Sales after 1900
            "listing_date": datetime(1900, 1, 1),    # Listings after 1900
            "import_date": datetime(2020, 1, 1)      # Imports after 2020
        }
        
        date_errors = {}
        current_date = datetime.now()
        
        for field, min_date in date_fields.items():
            if field not in data.columns:
                continue
                
            # Convert to datetime, coerce errors to NaN
            date_values = pd.to_datetime(data[field], errors='coerce')
            
            # Count invalid dates (before min_date or in the future)
            invalid_dates = ((date_values < min_date) | (date_values > current_date)) & date_values.notna()
            error_count = invalid_dates.sum()
            
            if error_count > 0:
                date_errors[field] = {
                    "min_date": min_date.isoformat(),
                    "max_date": current_date.isoformat(),
                    "error_count": int(error_count),
                    "error_percentage": round((error_count / len(data)) * 100, 2)
                }
        
        if date_errors:
            return {
                "status": "failed",
                "message": "Date validation failed for some fields",
                "errors": date_errors
            }
        else:
            return {
                "status": "passed",
                "message": "All dates are valid and within reasonable ranges"
            }
    
    def validate_duplicates(self, data):
        """
        Check for duplicate properties in the data.
        
        Args:
            data (pd.DataFrame): Property data
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating for duplicates")
        
        duplicate_checks = {}
        
        # Check for exact duplicates (all columns)
        exact_duplicates = data.duplicated().sum()
        duplicate_checks["exact_duplicates"] = {
            "count": int(exact_duplicates),
            "percentage": round((exact_duplicates / len(data)) * 100, 2)
        }
        
        # Check for address duplicates
        if "address" in data.columns and "city" in data.columns and "state" in data.columns and "zip_code" in data.columns:
            address_columns = ["address", "city", "state", "zip_code"]
            address_duplicates = data.duplicated(subset=address_columns, keep='first').sum()
            duplicate_checks["address_duplicates"] = {
                "count": int(address_duplicates),
                "percentage": round((address_duplicates / len(data)) * 100, 2)
            }
        
        # Check for parcel ID duplicates
        if "parcel_id" in data.columns:
            parcel_duplicates = data.duplicated(subset=["parcel_id"], keep='first').sum()
            duplicate_checks["parcel_id_duplicates"] = {
                "count": int(parcel_duplicates),
                "percentage": round((parcel_duplicates / len(data)) * 100, 2)
            }
        
        # Determine status based on threshold
        threshold = 5  # 5% duplicate rate is considered a failure
        failed = any(check["percentage"] > threshold for check in duplicate_checks.values())
        
        if failed:
            return {
                "status": "failed",
                "message": f"Duplicate validation failed (threshold: {threshold}%)",
                "checks": duplicate_checks
            }
        else:
            return {
                "status": "passed",
                "message": "Duplicate validation passed",
                "checks": duplicate_checks
            }
    
    def validate_cross_source_consistency(self, data):
        """
        Check for consistency across different data sources.
        
        Args:
            data (pd.DataFrame): Property data
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating cross-source consistency")
        
        if "data_source" not in data.columns:
            return {
                "status": "skipped",
                "message": "Data source information is missing"
            }
        
        # Get unique data sources
        sources = data["data_source"].unique()
        
        if len(sources) < 2:
            return {
                "status": "skipped",
                "message": "Multiple data sources are required for cross-source validation"
            }
        
        # Identify properties that appear in multiple sources
        if "address" in data.columns and "city" in data.columns and "state" in data.columns and "zip_code" in data.columns:
            address_columns = ["address", "city", "state", "zip_code"]
            address_counts = data.groupby(address_columns)["data_source"].nunique()
            multi_source_properties = address_counts[address_counts > 1].reset_index()
            
            if len(multi_source_properties) == 0:
                return {
                    "status": "skipped",
                    "message": "No properties found in multiple data sources"
                }
            
            # Check consistency of key numeric attributes across sources
            consistency_checks = {}
            key_attributes = ["square_feet", "year_built", "bedrooms", "bathrooms"]
            
            for attr in key_attributes:
                if attr not in data.columns:
                    continue
                
                # Calculate variation across sources for the same property
                attr_variations = []
                
                for _, row in multi_source_properties.iterrows():
                    # Create filter for this property
                    property_filter = True
                    for col in address_columns:
                        property_filter = property_filter & (data[col] == row[col])
                    
                    # Get values for this property across sources
                    property_data = data[property_filter]
                    attr_values = pd.to_numeric(property_data[attr], errors='coerce')
                    
                    if attr_values.notna().sum() >= 2:
                        # Calculate variation coefficient
                        variation = attr_values.std() / attr_values.mean() if attr_values.mean() != 0 else 0
                        attr_variations.append(variation)
                
                if attr_variations:
                    avg_variation = np.mean(attr_variations)
                    consistency_checks[attr] = {
                        "average_variation": round(avg_variation, 2),
                        "property_count": len(attr_variations)
                    }
            
            # Define threshold for consistency issues
            threshold = 0.10  # 10% variation is considered inconsistent
            failed = any(check["average_variation"] > threshold for check in consistency_checks.values())
            
            if failed:
                return {
                    "status": "failed",
                    "message": f"Cross-source consistency issues detected (threshold: {threshold})",
                    "checks": consistency_checks,
                    "multi_source_count": len(multi_source_properties)
                }
            else:
                return {
                    "status": "passed",
                    "message": "Cross-source consistency validation passed",
                    "checks": consistency_checks,
                    "multi_source_count": len(multi_source_properties)
                }
        else:
            return {
                "status": "skipped",
                "message": "Address information is incomplete for cross-source validation"
            }
    
    def report_validation_results(self, results):
        """
        Log and report validation results.
        
        Args:
            results (dict): Validation results
        """
        overall_status = results["status"]
        validations = results["validations"]
        
        logger.info(f"Data validation completed with overall status: {overall_status}")
        
        # Log detailed results for each validation check
        for validation_name, validation_result in validations.items():
            status = validation_result["status"]
            message = validation_result["message"]
            
            if status == "passed":
                logger.info(f"✓ {validation_name}: {message}")
            elif status == "failed":
                logger.warning(f"✗ {validation_name}: {message}")
            else:
                logger.info(f"- {validation_name}: {message}")
                
            # Log additional details for failed validations
            if status == "failed":
                if "errors" in validation_result:
                    for field, error in validation_result["errors"].items():
                        logger.warning(f"  - Field '{field}': {error}")
                elif "failed_fields" in validation_result:
                    for field, value in validation_result["failed_fields"].items():
                        logger.warning(f"  - Field '{field}': {value}%")
                elif "checks" in validation_result:
                    for check_name, check in validation_result["checks"].items():
                        if isinstance(check, dict) and "error_percentage" in check:
                            logger.warning(f"  - Check '{check_name}': {check['error_percentage']}%")
                        elif isinstance(check, dict) and "percentage" in check:
                            logger.warning(f"  - Check '{check_name}': {check['percentage']}%")
        
        # Store validation results in the database
        self.db.store_validation_results(results)
