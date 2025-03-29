"""
Test script for the validate_property_data function.
This script tests the standalone property data validation function.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from etl.data_validation import validate_property_data

def create_test_data():
    """Create a test DataFrame with various validation issues."""
    # Create a sample DataFrame with property data
    data = {
        # ID fields with some duplicates
        'property_id': ['P001', 'P002', 'P003', 'P003', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010'],
        'parcel_id': ['PA001', 'PA002', 'PA003', 'PA004', 'PA005', 'PA005', 'PA007', 'PA008', 'PA009', 'PA010'],
        
        # Address information
        'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '101 Cedar Ln', '202 Maple Dr', 
                   '303 Birch Blvd', '404 Elm St', '505 Walnut Ave', '606 Cherry Ln', '707 Ash Rd'],
        'city': ['Anytown', 'Sometown', 'Anytown', 'Othertown', 'Sometown', 
                'Othertown', 'Anytown', 'Sometown', 'Othertown', 'Anytown'],
        'state': ['CA', 'CA', 'CA', 'NY', 'NY', 'TX', 'TX', 'FL', 'FL', 'CA'],
        'zip_code': ['12345', '12346', '12345', '23456', '23457', '34567', '34568', '45678', '45679', '12347'],
        
        # Property characteristics - some numeric fields with issues
        'property_type': ['Single Family', 'Single Family', 'Condo', 'Single Family', 'Townhouse', 
                         'Single Family', 'Condo', 'Single Family', 'Townhouse', 'Single Family'],
        'square_feet': [1500, 2200, 900, 3000, 1800, 2500, 1000, "Invalid", 1700, 2800],
        'lot_size': [0.25, 0.5, 0.1, 1.2, 0.3, 0.6, 0.005, 0.4, 0.35, 150],  # One below and one above range
        'bedrooms': [3, 4, 2, 5, 3, 4, 2, 3, 3, 25],  # One above range
        'bathrooms': [2, 2.5, 1, 3.5, 2, 2.5, 1.5, 2, 2, 2.5],
        
        # Financial information
        'list_price': [500000, 750000, 350000, 1200000, 650000, 850000, 400000, 700000, 550000, 950000],
        'last_sale_price': [480000, 720000, 340000, 1100000, 625000, 820000, 390000, 680000, 530000, 900],  # One below range
        
        # Dates - some with format issues and future dates
        'last_sale_date': ['2022-01-15', '2021-11-20', '2022-03-05', '2021-09-10', '2022-02-28', 
                          '2021-10-15', '2022-04-10', '2021-12-05', 'invalid-date', '2022-05-20'],
        'listing_date': ['2022-05-01', '2022-04-15', '2022-06-10', '2022-03-20', '2022-05-15', 
                        '2022-04-05', '2022-06-20', '2022-03-10', '2022-05-25', (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')],  # One future date
    }
    
    return pd.DataFrame(data)

def main():
    """Run the validation test and print results."""
    print("Creating test data for property validation...")
    test_data = create_test_data()
    print(f"Created test DataFrame with {len(test_data)} rows")
    
    print("\nRunning property data validation...")
    validation_passed, validation_results = validate_property_data(test_data)
    
    print(f"\nValidation passed: {validation_passed}")
    print("\nValidation Summary:")
    
    # Format the results for better readability
    for category, result in validation_results['categories'].items():
        status = result['status']
        status_symbol = "✓" if status == "passed" else "✗" if status == "failed" else "⚠"
        
        print(f"{status_symbol} {category}: {status}")
        
        if status == "failed":
            for issue in result['issues']:
                if 'field' in issue:
                    field = issue['field']
                    if 'duplicate_count' in issue:
                        print(f"  - Field '{field}': {issue['duplicate_count']} duplicates ({issue['duplicate_percentage']}%)")
                    elif 'invalid_count' in issue:
                        print(f"  - Field '{field}': {issue['invalid_count']} invalid format ({issue['invalid_percentage']}%)")
                    elif 'future_dates_count' in issue:
                        print(f"  - Field '{field}': {issue['future_dates_count']} future dates ({issue['future_percentage']}%)")
                    elif 'issue_type' in issue:
                        if issue['issue_type'] == 'non_numeric':
                            print(f"  - Field '{field}': {issue['count']} non-numeric values ({issue['percentage']}%)")
                        elif issue['issue_type'] == 'below_minimum':
                            print(f"  - Field '{field}': {issue['count']} values below minimum {issue['min_value']} ({issue['percentage']}%)")
                        elif issue['issue_type'] == 'above_maximum':
                            print(f"  - Field '{field}': {issue['count']} values above maximum {issue['max_value']} ({issue['percentage']}%)")
                elif 'message' in issue:
                    print(f"  - {issue['message']}")
    
    # Save detailed results to a file
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    print("\nDetailed results saved to validation_results.json")

if __name__ == "__main__":
    main()