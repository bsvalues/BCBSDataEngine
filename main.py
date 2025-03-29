"""
Main script for the BCBS_Values ETL process.
Orchestrates the extraction, transformation, and loading of real estate data.
Also exports the Flask app for gunicorn to use.
"""
import os
import logging
import argparse
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
from etl.data_validation import DataValidator
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
        "--batch-size", 
        type=int, 
        default=int(os.getenv("ETL_BATCH_SIZE", 100)),
        help="Number of records to process in each batch"
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
                mls_data = mls_scraper.extract()
                mls_scraper.transform_and_load(mls_data, db)
                
            if "narrpr" in sources:
                logger.info("Starting NARRPR data extraction")
                narrpr_scraper = NARRPRScraper(batch_size=args.batch_size)
                narrpr_data = narrpr_scraper.extract()
                narrpr_scraper.transform_and_load(narrpr_data, db)
                
            if "pacs" in sources:
                logger.info("Starting PACS data extraction")
                pacs_importer = PACSImporter(batch_size=args.batch_size)
                pacs_data = pacs_importer.extract()
                pacs_importer.transform_and_load(pacs_data, db)
        
        # Always run data validation
        logger.info("Starting data validation")
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
