"""
NARRPR (National Association of REALTORS® Realtors Property Resource) data extraction module.
Handles extraction of property data from NARRPR API and web scraping.
"""
import os
import re
import time
import json
import logging
import random
import pandas as pd
from datetime import datetime

# Selenium-related imports for web scraping
import selenium
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class NARRPRScraper:
    """
    Scraper for extracting data from NARRPR (National Association of REALTORS® Realtors Property Resource).
    """
    
    def __init__(self, batch_size=100):
        """
        Initialize the NARRPR scraper.
        
        Args:
            batch_size (int): Number of records to fetch in each API call
        """
        self.batch_size = batch_size
        self.api_key = os.environ.get('NARRPR_API_KEY')
        
        if not self.api_key:
            logger.warning("NARRPR_API_KEY not found in environment variables")
    
    def extract(self, location=None, property_type=None):
        """
        Extract property data from NARRPR API.
        
        Args:
            location (str, optional): Location (city, zip code) to search for properties
            property_type (str, optional): Type of property to search for
            
        Returns:
            pd.DataFrame: DataFrame containing extracted property data
        """
        # Implementation for API-based extraction
        logger.info("NARRPR API extraction not yet implemented")
        logger.info("Use the Selenium-based scraper with --narrpr-use-selenium flag instead")
        
        # Return empty DataFrame for now
        return pd.DataFrame()
    
    def _make_api_request(self, endpoint, params):
        """
        Make a request to the NARRPR API.
        
        Args:
            endpoint (str): API endpoint to call
            params (dict): Parameters for the API request
            
        Returns:
            dict: JSON response from the API
        """
        # Placeholder for API request implementation
        logger.info(f"NARRPR API request to {endpoint} with params {params}")
        return {}
    
    def transform_and_load(self, data, db):
        """
        Transform NARRPR data and load it into the database.
        
        Args:
            data (pd.DataFrame): DataFrame containing NARRPR property data
            db: Database connection object
            
        Returns:
            int: Number of records loaded into the database
        """
        if data.empty:
            logger.warning("No NARRPR data to transform and load")
            return 0
        
        logger.info(f"Transforming {len(data)} NARRPR records")
        
        # Transform the data
        transformed_data = self._transform_data(data)
        
        # Load the data into the database
        records_loaded = self._load_data(transformed_data, db)
        
        return records_loaded
    
    def _transform_data(self, data):
        """
        Transform NARRPR data into the format required for database.
        
        Args:
            data (pd.DataFrame): Raw NARRPR data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # Make a copy to avoid modifying the original
        transformed = data.copy()
        
        # Add data source and import date
        transformed['data_source'] = 'NARRPR'
        transformed['import_date'] = datetime.now()
        
        # Standardize column names if needed
        column_mapping = {
            'scrape_date': 'import_date',
            'sqft': 'square_feet',
            'price': 'list_price',
            'bath': 'bathrooms',
            'bed': 'bedrooms',
            'zip': 'zip_code',
            'build_year': 'year_built'
        }
        
        # Rename columns that exist in the DataFrame
        for old_col, new_col in column_mapping.items():
            if old_col in transformed.columns and new_col not in transformed.columns:
                transformed.rename(columns={old_col: new_col}, inplace=True)
        
        # Log data types for debugging
        logger.debug(f"Data types after transformation: {transformed.dtypes}")
        
        return transformed
    
    def _load_data(self, data, db):
        """
        Load transformed data into the database.
        
        Args:
            data (pd.DataFrame): Transformed NARRPR data
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
            records_loaded = db.insert_properties(data, source='NARRPR')
            logger.info(f"Loaded {records_loaded} NARRPR records into database")
            return records_loaded
        except Exception as e:
            logger.error(f"Error loading NARRPR data into database: {str(e)}")
            raise
    
    def narrpr_login_and_scrape(self, search_location='', property_type='residential', output_path=None, max_results=100):
        """
        Login to NARRPR website using Selenium and scrape property data.
        
        Args:
            search_location (str): Location to search for properties (city, zip code, etc.)
            property_type (str): Type of property to search for ('residential', 'commercial', etc.)
            output_path (str, optional): Path to save the CSV file. If None, creates a timestamped file
                                         in the current directory.
            max_results (int): Maximum number of property results to scrape.
            
        Returns:
            pd.DataFrame: DataFrame containing the scraped property data
            
        Raises:
            ValueError: If login credentials are missing or invalid
            WebDriverException: If there's an issue with the browser automation
        """
        # Get credentials from environment variables
        username = os.environ.get('NARRPR_USERNAME')
        password = os.environ.get('NARRPR_PASSWORD')
        
        # Check if credentials are available
        if not username or not password:
            logger.error("NARRPR login credentials not found in environment variables")
            raise ValueError("Missing NARRPR login credentials")
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        properties_data = []
        driver = None
        
        try:
            # Initialize the driver
            logger.info("Initializing Chrome driver for NARRPR scraping")
            service = Service(ChromeDriverManager().install())
            driver = selenium.webdriver.Chrome(service=service, options=chrome_options)
            
            # Navigate to NARRPR login page
            login_url = "https://www.narrpr.com/login"
            logger.info(f"Navigating to NARRPR login page: {login_url}")
            driver.get(login_url)
            
            # Wait for the page to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            
            # Login
            logger.info(f"Logging in as {username}")
            driver.find_element(By.ID, "username").send_keys(username)
            driver.find_element(By.ID, "password").send_keys(password)
            driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
            
            # Wait for login to complete and dashboard to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".dashboard-container"))
            )
            logger.info("Successfully logged in to NARRPR")
            
            # Navigate to property search
            search_url = "https://www.narrpr.com/search"
            logger.info(f"Navigating to search page: {search_url}")
            driver.get(search_url)
            
            # Wait for search page to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "location-search"))
            )
            
            # Enter search location
            logger.info(f"Searching for properties in {search_location}")
            location_input = driver.find_element(By.ID, "location-search")
            location_input.clear()
            location_input.send_keys(search_location)
            
            # Select property type
            property_type_select = driver.find_element(By.ID, "property-type-select")
            property_type_select.click()
            
            # Wait for dropdown options to appear
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, f"option[value='{property_type}']"))
            )
            
            # Select the property type option
            driver.find_element(By.CSS_SELECTOR, f"option[value='{property_type}']").click()
            
            # Submit search
            driver.find_element(By.ID, "search-button").click()
            
            # Wait for search results to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".property-results"))
            )
            
            # Get search result count
            result_count_element = driver.find_element(By.CSS_SELECTOR, ".result-count")
            result_count_text = result_count_element.text
            result_count_match = re.search(r'(\d+)', result_count_text)
            result_count = int(result_count_match.group(1)) if result_count_match else 0
            
            logger.info(f"Found {result_count} properties in {search_location}")
            
            # Limit the number of results to scrape
            results_to_scrape = min(result_count, max_results)
            
            # Scrape the property cards
            property_cards = driver.find_elements(By.CSS_SELECTOR, ".property-card")
            
            for i, card in enumerate(property_cards):
                if i >= results_to_scrape:
                    break
                    
                try:
                    # Extract property details from card
                    address = card.find_element(By.CSS_SELECTOR, ".property-address").text
                    price_element = card.find_element(By.CSS_SELECTOR, ".property-price")
                    price_text = price_element.text.replace('$', '').replace(',', '')
                    list_price = float(price_text) if price_text.strip() else None
                    
                    # Extract property features
                    features = card.find_element(By.CSS_SELECTOR, ".property-features").text
                    
                    # Parse features (typically in format like "3 bd • 2 ba • 1,500 sqft")
                    bedrooms = None
                    bathrooms = None
                    square_feet = None
                    
                    # Extract bedrooms
                    bed_match = re.search(r'(\d+)\s*bd', features)
                    if bed_match:
                        bedrooms = float(bed_match.group(1))
                    
                    # Extract bathrooms
                    bath_match = re.search(r'(\d+\.?\d*)\s*ba', features)
                    if bath_match:
                        bathrooms = float(bath_match.group(1))
                    
                    # Extract square footage
                    sqft_match = re.search(r'([\d,]+)\s*sqft', features)
                    if sqft_match:
                        square_feet = float(sqft_match.group(1).replace(',', ''))
                    
                    # Get city, state, zip from address
                    address_parts = address.split(',')
                    street_address = address_parts[0].strip()
                    
                    city = None
                    state = None
                    zip_code = None
                    
                    if len(address_parts) > 1:
                        location_parts = address_parts[1].strip().split()
                        if len(location_parts) >= 2:
                            city = ' '.join(location_parts[:-2])
                            state = location_parts[-2]
                            zip_code = location_parts[-1]
                    
                    # Click on the property card to get more details
                    card.click()
                    
                    # Wait for property details page to load
                    WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".property-details"))
                    )
                    
                    # Extract additional details
                    property_id = None
                    property_type_detail = property_type  # Default to the search property type
                    year_built = None
                    lot_size = None
                    
                    # Property ID
                    try:
                        property_id_element = driver.find_element(By.CSS_SELECTOR, ".property-id")
                        property_id = property_id_element.text.split(':')[-1].strip()
                    except NoSuchElementException:
                        pass
                    
                    # Property Type (more specific)
                    try:
                        property_type_element = driver.find_element(By.CSS_SELECTOR, ".property-type")
                        property_type_detail = property_type_element.text
                    except NoSuchElementException:
                        pass
                    
                    # Year Built
                    try:
                        year_built_element = driver.find_element(By.CSS_SELECTOR, ".year-built")
                        year_built_text = year_built_element.text.split(':')[-1].strip()
                        year_built = int(year_built_text) if year_built_text.isdigit() else None
                    except (NoSuchElementException, ValueError):
                        pass
                    
                    # Lot Size
                    try:
                        lot_size_element = driver.find_element(By.CSS_SELECTOR, ".lot-size")
                        lot_size_text = lot_size_element.text
                        lot_size_match = re.search(r'([\d.]+)', lot_size_text)
                        lot_size = float(lot_size_match.group(1)) if lot_size_match else None
                    except (NoSuchElementException, ValueError):
                        pass
                    
                    # Create property data dictionary
                    property_data = {
                        'property_id': property_id,
                        'address': street_address,
                        'city': city,
                        'state': state,
                        'zip_code': zip_code,
                        'list_price': list_price,
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'square_feet': square_feet,
                        'property_type': property_type_detail,
                        'year_built': year_built,
                        'lot_size': lot_size,
                        'data_source': 'NARRPR',
                        'scrape_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    properties_data.append(property_data)
                    logger.info(f"Scraped property {i+1}/{results_to_scrape}: {street_address}")
                    
                    # Go back to results page
                    driver.back()
                    
                    # Wait for results to load again
                    WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".property-results"))
                    )
                    
                    # Re-fetch the property cards (they might have become stale after navigation)
                    property_cards = driver.find_elements(By.CSS_SELECTOR, ".property-card")
                    
                    # Add a random delay to avoid rate limiting
                    time.sleep(random.uniform(1.0, 3.0))
                    
                except Exception as e:
                    logger.error(f"Error scraping property {i+1}: {str(e)}")
                    # Continue with next property
                    continue
        
        except TimeoutException as e:
            logger.error(f"Timeout while scraping NARRPR: {str(e)}")
            raise
        except WebDriverException as e:
            logger.error(f"WebDriver error while scraping NARRPR: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during NARRPR scraping: {str(e)}")
            raise
        finally:
            # Close the driver if it was initialized
            if driver:
                logger.info("Closing Chrome driver")
                driver.quit()
        
        # Check if we scraped any properties
        if not properties_data:
            logger.warning("No property data was scraped")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(properties_data)
        
        # Save to CSV if path is provided
        if output_path:
            logger.info(f"Saving scraped data to {output_path}")
            df.to_csv(output_path, index=False)
        
        return df
    
    def scrape_and_load(self, search_location, property_type, db, output_path=None, max_results=100):
        """
        Combined method to scrape NARRPR data and load it into the database.
        
        Args:
            search_location (str): Location to search for properties (city, zip code, etc.)
            property_type (str): Type of property to search for ('residential', 'commercial', etc.)
            db: Database connection object
            output_path (str, optional): Path to save the CSV file
            max_results (int): Maximum number of property results to scrape
            
        Returns:
            int: Number of records loaded into the database
        """
        logger.info(f"Starting NARRPR scrape and load process for {search_location}")
        
        try:
            # Scrape data from NARRPR
            data = self.narrpr_login_and_scrape(
                search_location=search_location,
                property_type=property_type,
                output_path=output_path,
                max_results=max_results
            )
            
            # If no data was scraped, return 0
            if data.empty:
                logger.warning("No data was scraped from NARRPR website")
                return 0
            
            # Transform and load the data
            records_loaded = self.transform_and_load(data, db)
            
            return records_loaded
            
        except Exception as e:
            logger.error(f"Error during NARRPR scrape and load process: {str(e)}")
            raise