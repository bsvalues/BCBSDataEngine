"""
Main script for the BCBS_Values web application.

This script initializes logging configuration and provides the entry
point for both the web application and the ETL pipeline.
"""
import os
import sys
import argparse
import time
from datetime import datetime

# Setup logging first thing so that all modules can use it
from utils.logging_config import setup_logging, get_etl_logger, log_data_operation, log_validation_result

# Initialize the root logger with basic configuration
logger = setup_logging(
    log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    log_file=f"bcbs_values_{datetime.now().strftime('%Y%m%d')}.log"
)

# Log application startup
logger.info("==== BCBS_Values Application Starting ====")
logger.info(f"Python version: {sys.version}")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")

# Add the current directory to the Python path if needed
if "." not in sys.path:
    sys.path.append(".")
    logger.debug("Added current directory to Python path")

# Import application modules after logging is configured
from app import app
from etl import data_validation

def run_etl_pipeline(sources=None, validate_only=False):
    """
    Run the ETL pipeline for specified data sources.
    
    Args:
        sources (list): List of data sources to process ('pacs', 'mls', 'narrpr', 'all')
        validate_only (bool): If True, only run validation without data loading
    """
    # Get a dedicated logger for the ETL process
    etl_logger = get_etl_logger()
    etl_logger.info(f"Starting ETL pipeline - sources: {sources}, validate_only: {validate_only}")
    
    # Map sources to modules
    source_modules = {
        'pacs': 'etl.pacs_import',
        'mls': 'etl.mls_scraper',
        'narrpr': 'etl.narrpr_scraper',
    }
    
    # If 'all' is specified, process all sources
    if sources and 'all' in sources:
        sources = list(source_modules.keys())
    
    # Track overall results
    results = {
        'total_records_processed': 0,
        'total_records_loaded': 0,
        'validation_results': {},
        'errors': [],
        'start_time': time.time(),
    }
    
    try:
        # Process each requested source
        for source in sources:
            if source not in source_modules:
                etl_logger.warning(f"Unknown source: {source}, skipping")
                continue
                
            etl_logger.info(f"Processing source: {source}")
            start_time = time.time()
            
            try:
                # Import the appropriate module
                module_name = source_modules[source]
                module = __import__(module_name, fromlist=[''])
                
                # Extract data
                etl_logger.info(f"Extracting data from {source}")
                data = module.extract_data()
                extract_duration = (time.time() - start_time) * 1000  # milliseconds
                
                # Log the extraction operation
                log_data_operation(
                    etl_logger, 
                    "extract", 
                    source, 
                    len(data) if hasattr(data, '__len__') else 0,
                    extract_duration
                )
                
                # Transform data
                transform_start = time.time()
                etl_logger.info(f"Transforming data from {source}")
                transformed_data = module.transform_data(data)
                transform_duration = (time.time() - transform_start) * 1000
                
                # Log the transformation operation
                log_data_operation(
                    etl_logger, 
                    "transform", 
                    source, 
                    len(transformed_data) if hasattr(transformed_data, '__len__') else 0,
                    transform_duration
                )
                
                # Validate data
                validation_start = time.time()
                etl_logger.info(f"Validating data from {source}")
                validation_results = data_validation.validate_data(transformed_data, source)
                validation_duration = (time.time() - validation_start) * 1000
                
                # Log validation results
                log_validation_result(etl_logger, validation_results)
                
                # Store validation results for reporting
                results['validation_results'][source] = validation_results
                
                # Update overall counts
                records_processed = validation_results.get('total_records', 0)
                results['total_records_processed'] += records_processed
                
                # Load data if not in validate-only mode and validation passed
                if not validate_only and validation_results.get('validation_passed', False):
                    load_start = time.time()
                    etl_logger.info(f"Loading data from {source}")
                    records_loaded = module.load_data(transformed_data)
                    load_duration = (time.time() - load_start) * 1000
                    
                    # Log the load operation
                    log_data_operation(
                        etl_logger, 
                        "load", 
                        source, 
                        records_loaded,
                        load_duration
                    )
                    
                    results['total_records_loaded'] += records_loaded
                    etl_logger.info(f"Successfully loaded {records_loaded} records from {source}")
                else:
                    if validate_only:
                        etl_logger.info(f"Skipping data load for {source} (validate-only mode)")
                    else:
                        etl_logger.warning(f"Skipping data load for {source} due to validation failures")
                
            except Exception as e:
                etl_logger.error(f"Error processing source {source}: {str(e)}", exc_info=True)
                results['errors'].append({
                    'source': source,
                    'error': str(e),
                    'type': str(type(e).__name__)
                })
    finally:
        # Calculate overall duration
        results['duration_seconds'] = time.time() - results['start_time']
        
        # Log summary
        etl_logger.info(f"ETL pipeline completed in {results['duration_seconds']:.2f} seconds")
        etl_logger.info(f"Total records processed: {results['total_records_processed']}")
        
        if not validate_only:
            etl_logger.info(f"Total records loaded: {results['total_records_loaded']}")
        
        if results['errors']:
            etl_logger.error(f"Encountered {len(results['errors'])} errors during ETL process")
        
        # Save validation results to file
        try:
            import json
            from pathlib import Path
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"validation_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            etl_logger.info(f"Saved validation results to {results_file}")
        except Exception as e:
            etl_logger.error(f"Error saving validation results: {str(e)}")
    
    return results

# Main entry point
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BCBS_Values Application")
    parser.add_argument("--sources", nargs="+", choices=['pacs', 'mls', 'narrpr', 'all'], 
                        help="Data sources to process")
    parser.add_argument("--validate-only", action="store_true", 
                        help="Only validate data without loading it")
    parser.add_argument("--web", action="store_true", 
                        help="Start the web application")
    
    args = parser.parse_args()
    
    # Run ETL pipeline if sources are specified
    if args.sources:
        logger.info(f"Running ETL pipeline with sources: {args.sources}")
        run_etl_pipeline(args.sources, args.validate_only)
    
    # Start web application if requested or if no specific action was specified
    if args.web or not (args.sources):
        logger.info("Starting web application")
        app.run(host="0.0.0.0", port=5001, debug=True)