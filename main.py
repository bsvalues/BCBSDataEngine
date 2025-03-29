"""
Main script for the BCBS_Values ETL process.
Orchestrates the extraction, transformation, and loading of real estate data.
Also exports the Flask app for gunicorn to use.
"""
import os
import logging
import argparse
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from app import app

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("ETL_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    return parser.parse_args()


def run_etl_pipeline(args):
    """Run the ETL pipeline with the specified configuration."""
    logger.info("Initializing ETL pipeline")
    
    # Initialize database connection
    db = Database()
    
    try:
        # Process data sources based on arguments
        sources = ["mls", "narrpr", "pacs"] if "all" in args.sources else args.sources
        
        # Skip data extraction if only validation is requested
        if not args.validate_only:
            if "mls" in sources:
                logger.info("Starting MLS data extraction")
                mls_scraper = MLSScraper(batch_size=args.batch_size)
                # Use CSV file if provided, otherwise use API
                mls_data = mls_scraper.extract(csv_file_path=args.mls_csv)
                mls_scraper.transform_and_load(mls_data, db)
                
            if "narrpr" in sources:
                logger.info("Starting NARRPR data extraction")
                narrpr_scraper = NARRPRScraper(batch_size=args.batch_size)
                
                # Use CSV file if provided (bypasses API and Selenium scraping)
                if args.narrpr_csv:
                    logger.info(f"Loading NARRPR data from CSV file: {args.narrpr_csv}")
                    try:
                        # Load data from CSV
                        narrpr_data = pd.read_csv(args.narrpr_csv)
                        narrpr_scraper.transform_and_load(narrpr_data, db)
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
                    csv_path = f"narrpr_properties_{timestamp}.csv"
                    
                    # Call the integrated scrape and load function
                    records_loaded = narrpr_scraper.scrape_and_load(
                        search_location=args.narrpr_location,
                        property_type=args.narrpr_property_type,
                        db=db,
                        output_path=csv_path,
                        max_results=args.narrpr_max_results
                    )
                    logger.info(f"NARRPR Selenium scraper loaded {records_loaded} records")
                
                # Default: Use API extraction
                else:
                    logger.info("Using NARRPR API to extract data")
                    narrpr_data = narrpr_scraper.extract()
                    narrpr_scraper.transform_and_load(narrpr_data, db)
                
            if "pacs" in sources:
                logger.info("Starting PACS data extraction")
                pacs_importer = PACSImporter(batch_size=args.batch_size)
                # Add CSV option for PACS (would need to implement in PACSImporter class)
                pacs_data = pacs_importer.extract(file_path=args.pacs_csv)
                pacs_importer.transform_and_load(pacs_data, db)
        
        # Always run data validation
        logger.info("Starting data validation")
        
        # Determine which validation method to use
        if args.standalone_validation:
            logger.info("Using standalone property validation function")
            # Get all properties from the database
            properties = db.get_all_properties()
            
            if properties.empty:
                logger.warning("No properties found in database for validation")
                logger.info("ETL pipeline completed successfully")
                return
                
            # Run the standalone validation function
            validation_passed, validation_summary = validate_property_data(properties)
            
            # Log the validation results
            logger.info(f"Standalone validation complete. Passed: {validation_passed}")
            if not validation_passed:
                logger.warning("Validation issues found:")
                for category, result in validation_summary['categories'].items():
                    if result["status"] == "failed":
                        logger.warning(f"- {category}: {len(result['issues'])} issues found")
                        
            # Store results to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"validation_results_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(validation_summary, f, indent=2)
            logger.info(f"Detailed validation results saved to {result_file}")
            
            # Store in database as JSON string
            db.store_validation_results({
                "status": "passed" if validation_passed else "failed",
                "results": json.dumps(validation_summary)
            })
        else:
            # Use the DataValidator class (original behavior)
            validator = DataValidator(db)
            validation_results = validator.validate_all()
            validator.report_validation_results(validation_results)
        
        logger.info("ETL pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    args = parse_arguments()
    run_etl_pipeline(args)
