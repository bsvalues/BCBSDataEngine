"""
Main script for the BCBS_Values ETL process.
Orchestrates the extraction, transformation, and loading of real estate data
from multiple sources (MLS, NARRPR, PACS) into a PostgreSQL database.
Also exports the Flask app for gunicorn to use.
"""
import os
import logging
import argparse
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
from app import app

# Load environment variables
load_dotenv()

# Configure logging with more detailed format and file output
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/etl_pipeline_{timestamp}.log"

logging.basicConfig(
    level=getattr(logging, os.getenv("ETL_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import ETL components
from etl.mls_scraper import MLSScraper
from etl.narrpr_scraper import NARRPRScraper
from etl.pacs_import import PACSImporter
from etl.data_validation import DataValidator, validate_property_data
from db.database import Database


def parse_arguments():
    """Parse command line arguments for the ETL process."""
    parser = argparse.ArgumentParser(description="BCBS_Values ETL Pipeline")
    parser.add_argument(
        "--sources", 
        nargs="+", 
        choices=["mls", "narrpr", "pacs", "all"], 
        default=["all"],
        help="Data sources to process (mls, narrpr, pacs, or all)"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate existing data without importing new data"
    )
    parser.add_argument(
        "--standalone-validation",
        action="store_true",
        help="Use the standalone property validation function instead of the DataValidator class"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=int(os.getenv("ETL_BATCH_SIZE", 100)),
        help="Number of records to process in each batch"
    )
    parser.add_argument(
        "--create-indexes",
        action="store_true",
        help="Create database indexes after loading data"
    )
    parser.add_argument(
        "--mls-csv", 
        type=str,
        help="Path to CSV file containing MLS data (bypasses API extraction)"
    )
    parser.add_argument(
        "--pacs-csv", 
        type=str,
        help="Path to CSV file containing PACS data (bypasses API extraction)"
    )
    parser.add_argument(
        "--narrpr-csv", 
        type=str,
        help="Path to CSV file containing NARRPR data (bypasses API extraction)"
    )
    parser.add_argument(
        "--narrpr-location",
        type=str,
        help="Location to search for NARRPR properties (city, zip code, etc.)"
    )
    parser.add_argument(
        "--narrpr-property-type",
        type=str,
        default="residential",
        choices=["residential", "commercial", "land", "multifamily"],
        help="Property type to search for in NARRPR"
    )
    parser.add_argument(
        "--narrpr-max-results",
        type=int,
        default=100,
        help="Maximum number of results to scrape from NARRPR"
    )
    parser.add_argument(
        "--narrpr-use-selenium",
        action="store_true",
        help="Use Selenium-based web scraping for NARRPR data instead of API"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="etl_outputs",
        help="Directory to store output files (validation results, etc.)"
    )
    return parser.parse_args()


def run_extraction_step(source, scraper_class, args, db):
    """
    Run the extraction step for a specific data source.
    
    Args:
        source (str): Data source name (mls, narrpr, pacs)
        scraper_class: Scraper/Importer class to instantiate
        args: Command-line arguments
        db: Database connection object
        
    Returns:
        pd.DataFrame: Combined data extracted from the source
    """
    start_time = time.time()
    logger.info(f"========== STARTING {source.upper()} EXTRACTION ==========")
    
    try:
        scraper = scraper_class(batch_size=args.batch_size)
        
        # Determine data source and extraction method
        if source == "mls":
            csv_path = args.mls_csv
            logger.info(f"Extracting MLS data {'from CSV: ' + csv_path if csv_path else 'from API'}")
            data = scraper.extract(csv_file_path=csv_path)
            
        elif source == "narrpr":
            # Use CSV file if provided (bypasses API and Selenium scraping)
            if args.narrpr_csv:
                csv_path = args.narrpr_csv
                logger.info(f"Loading NARRPR data from CSV file: {csv_path}")
                try:
                    # Load data from CSV
                    data = pd.read_csv(csv_path)
                except Exception as e:
                    logger.error(f"Failed to load NARRPR CSV data: {str(e)}")
                    raise
            
            # Use Selenium scraper if specified
            elif args.narrpr_use_selenium:
                if not args.narrpr_location:
                    logger.error("NARRPR location parameter is required when using Selenium scraper")
                    raise ValueError("--narrpr-location parameter is required with --narrpr-use-selenium")
                
                logger.info(f"Using Selenium to scrape NARRPR data for location: {args.narrpr_location}")
                
                # Generate timestamped CSV filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = args.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                csv_path = f"{output_dir}/narrpr_properties_{timestamp}.csv"
                
                # Call the integrated scrape and load function
                records_loaded = scraper.scrape_and_load(
                    search_location=args.narrpr_location,
                    property_type=args.narrpr_property_type,
                    db=db,
                    output_path=csv_path,
                    max_results=args.narrpr_max_results
                )
                logger.info(f"NARRPR Selenium scraper loaded {records_loaded} records")
                
                # Return empty DataFrame since the scraper loaded the data directly
                return pd.DataFrame()
            
            # Default: Use API extraction
            else:
                logger.info("Using NARRPR API to extract data")
                data = scraper.extract()
        
        elif source == "pacs":
            csv_path = args.pacs_csv
            logger.info(f"Extracting PACS data {'from CSV: ' + csv_path if csv_path else 'from API'}")
            data = scraper.extract(file_path=csv_path)
        
        else:
            logger.error(f"Unknown data source: {source}")
            return pd.DataFrame()
        
        # Log extraction summary
        if data is not None and not data.empty:
            logger.info(f"Successfully extracted {len(data)} records from {source.upper()}")
            logger.info(f"Sample columns: {', '.join(data.columns[:5])}...")
            
            # Calculate memory usage
            memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)  # In MB
            logger.info(f"DataFrame memory usage: {memory_usage:.2f} MB")
        else:
            logger.warning(f"No data extracted from {source.upper()}")
            
        return data
        
    except Exception as e:
        logger.error(f"{source.upper()} extraction failed: {str(e)}", exc_info=True)
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"========== COMPLETED {source.upper()} EXTRACTION in {elapsed_time:.2f} seconds ==========")


def run_transform_load_step(source, data, scraper, db):
    """
    Run the transform and load step for a specific data source.
    
    Args:
        source (str): Data source name (mls, narrpr, pacs)
        data (pd.DataFrame): Data to transform and load
        scraper: Scraper/Importer instance
        db: Database connection object
        
    Returns:
        int: Number of records loaded
    """
    if data is None or data.empty:
        logger.warning(f"No {source} data to transform and load")
        return 0
        
    start_time = time.time()
    logger.info(f"========== STARTING {source.upper()} TRANSFORM & LOAD ==========")
    
    try:
        logger.info(f"Transforming and loading {len(data)} records from {source.upper()}")
        records_loaded = scraper.transform_and_load(data, db)
        logger.info(f"Successfully loaded {records_loaded} records from {source.upper()} into the database")
        return records_loaded
        
    except Exception as e:
        logger.error(f"{source.upper()} transform and load failed: {str(e)}", exc_info=True)
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"========== COMPLETED {source.upper()} TRANSFORM & LOAD in {elapsed_time:.2f} seconds ==========")


def run_validation_step(db, args):
    """
    Run the validation step on the database data.
    
    Args:
        db: Database connection object
        args: Command-line arguments
        
    Returns:
        dict: Validation results
    """
    start_time = time.time()
    logger.info("========== STARTING DATA VALIDATION ==========")
    
    try:
        # Determine which validation method to use
        if args.standalone_validation:
            logger.info("Using standalone property validation function")
            
            # Get all properties from the database
            properties = db.get_all_properties()
            
            if properties.empty:
                logger.warning("No properties found in database for validation")
                return None
                
            # Run the standalone validation function
            validation_passed, validation_summary = validate_property_data(properties)
            
            # Log the validation results
            logger.info(f"Standalone validation complete. Passed: {validation_passed}")
            if not validation_passed:
                logger.warning("Validation issues found:")
                for category, result in validation_summary.get('categories', {}).items():
                    if result.get("status") == "failed":
                        logger.warning(f"- {category}: {len(result.get('issues', []))} issues found")
                        
            # Store results to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            result_file = f"{output_dir}/validation_results_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(validation_summary, f, indent=2)
            logger.info(f"Detailed validation results saved to {result_file}")
            
            # Store in database 
            db.store_validation_results({
                "status": "passed" if validation_passed else "failed",
                "results": json.dumps(validation_summary)
            })
            
            return validation_summary
        else:
            # Use the DataValidator class
            logger.info("Using DataValidator class for validation")
            validator = DataValidator(db)
            validation_results = validator.validate_all()
            validator.report_validation_results(validation_results)
            
            return validation_results
            
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}", exc_info=True)
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"========== COMPLETED DATA VALIDATION in {elapsed_time:.2f} seconds ==========")


def run_etl_pipeline(args):
    """Run the ETL pipeline with the specified configuration."""
    overall_start_time = time.time()
    logger.info("========== INITIALIZING ETL PIPELINE ==========")
    logger.info(f"Command-line arguments: {args}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize database connection
    db = Database()
    
    try:
        # Process data sources based on arguments
        sources = ["mls", "narrpr", "pacs"] if "all" in args.sources else args.sources
        logger.info(f"Processing data sources: {', '.join(sources)}")
        
        extraction_stats = {}
        load_stats = {}
        
        # Skip data extraction if only validation is requested
        if not args.validate_only:
            # MLS Data Extraction
            if "mls" in sources:
                mls_scraper = MLSScraper(batch_size=args.batch_size)
                mls_data = run_extraction_step("mls", MLSScraper, args, db)
                extraction_stats["mls"] = len(mls_data) if mls_data is not None else 0
                
                # Transform and load MLS data
                if mls_data is not None and not mls_data.empty:
                    records_loaded = run_transform_load_step("mls", mls_data, mls_scraper, db)
                    load_stats["mls"] = records_loaded
                
            # NARRPR Data Extraction
            if "narrpr" in sources:
                narrpr_scraper = NARRPRScraper(batch_size=args.batch_size)
                narrpr_data = run_extraction_step("narrpr", NARRPRScraper, args, db)
                extraction_stats["narrpr"] = len(narrpr_data) if narrpr_data is not None else 0
                
                # Transform and load NARRPR data (unless it was directly loaded by Selenium scraper)
                if narrpr_data is not None and not narrpr_data.empty:
                    records_loaded = run_transform_load_step("narrpr", narrpr_data, narrpr_scraper, db)
                    load_stats["narrpr"] = records_loaded
                
            # PACS Data Extraction
            if "pacs" in sources:
                pacs_importer = PACSImporter(batch_size=args.batch_size)
                pacs_data = run_extraction_step("pacs", PACSImporter, args, db)
                extraction_stats["pacs"] = len(pacs_data) if pacs_data is not None else 0
                
                # Transform and load PACS data
                if pacs_data is not None and not pacs_data.empty:
                    records_loaded = run_transform_load_step("pacs", pacs_data, pacs_importer, db)
                    load_stats["pacs"] = records_loaded
            
            # Create database indexes if requested
            if args.create_indexes:
                logger.info("Creating database indexes for improved query performance")
                try:
                    db.create_properties_indexes()
                    logger.info("Database indexes created successfully")
                except Exception as e:
                    logger.error(f"Failed to create database indexes: {str(e)}")
        
        # Run data validation
        validation_results = run_validation_step(db, args)
        
        # Log overall pipeline summary
        logger.info("========== ETL PIPELINE SUMMARY ==========")
        
        if not args.validate_only:
            logger.info("Extraction Statistics:")
            for source, count in extraction_stats.items():
                logger.info(f"  - {source.upper()}: {count} records extracted")
            
            logger.info("Load Statistics:")
            for source, count in load_stats.items():
                logger.info(f"  - {source.upper()}: {count} records loaded")
        
        validation_status = "Skipped"
        if validation_results:
            if isinstance(validation_results, dict) and "status" in validation_results:
                validation_status = validation_results["status"]
            elif isinstance(validation_results, tuple) and len(validation_results) > 0:
                validation_status = "Passed" if validation_results[0] else "Failed"
            else:
                validation_status = "Completed"
                
        logger.info(f"Validation Status: {validation_status}")
        
        overall_elapsed_time = time.time() - overall_start_time
        logger.info(f"Total runtime: {overall_elapsed_time:.2f} seconds")
        logger.info("ETL pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Closing database connection")
        db.close()


if __name__ == "__main__":
    args = parse_arguments()
    run_etl_pipeline(args)
