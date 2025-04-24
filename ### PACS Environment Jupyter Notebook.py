### PACS Environment Jupyter Notebook

```python
# Import Required Libraries
import pyodbc
import pandas as pd
from arcgis.gis import GIS
import openpyxl
import json

# Configuration for Database Connection
DB_CONFIG = {
    'DSN': 'chpacs.dsn',  # Data Source Name
    'DATABASE': 'pacs_training',
    'USER': '',
    'PASSWORD': '',
    'DRIVER': '{ODBC Driver 17 for SQL Server}'
}

# Function to Establish Database Connection
def connect_to_db():
    connection_string = (
        f"DSN={DB_CONFIG['DSN']};"
        f"DATABASE={DB_CONFIG['DATABASE']};"
        f"UID={DB_CONFIG['USER']};"
        f"PWD={DB_CONFIG['PASSWORD']};"
    )
    conn = pyodbc.connect(connection_string)
    return conn

# Example Query Execution
def execute_query(query):
    conn = connect_to_db()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

# Query: Fetch Tax District Data
query_tax_districts = "SELECT * FROM TaxDistricts"
data_tax_districts = execute_query(query_tax_districts)

# Display Data
print("Tax District Data:")
display(data_tax_districts)

# Export to Excel
output_path = "tax_district_data.xlsx"
data_tax_districts.to_excel(output_path, index=False)
print(f"Data exported to {output_path}")

# GIS Integration Example
# Authenticate with ArcGIS Online
GIS_CONFIG = {
    'URL': 'https://www.arcgis.com',
    'USER': '<username>',
    'PASSWORD': '<password>'
}

def gis_authenticate():
    gis = GIS(GIS_CONFIG['URL'], GIS_CONFIG['USER'], GIS_CONFIG['PASSWORD'])
    return gis

# Add Data to GIS
def upload_shapefile(file_path, title):
    gis = gis_authenticate()
    gis.content.add({"type": "Shapefile", "title": title}, data=file_path)

# Example: Upload Tax District Shapefile
shapefile_path = "tax_districts.zip"
shapefile_title = "Tax Districts"
# Uncomment to execute
# upload_shapefile(shapefile_path, shapefile_title)

# Automating Levy Calculations (Example Function)
def calculate_highest_lawful_levy(tax_year):
    df = execute_query(f"SELECT * FROM LevyData WHERE TaxYear = {tax_year}")
    df['HighestLawfulLevy'] = df['LastYearLevy'] * 1.01  # Example Calculation
    output_path = f"levy_calculation_{tax_year}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Levy calculations exported to {output_path}")

# Example Execution
calculate_highest_lawful_levy(2023)

# Load Stored Procedures and Foreign Key Relationships
with open("pacs_oltp_Procedures.json", "r") as proc_file:
    procedures = json.load(proc_file)
    print(f"Loaded {len(procedures)} procedures from PACS database.")

with open("pacs_oltp_Foreign_Keys.json", "r") as fk_file:
    foreign_keys = json.load(fk_file)
    print(f"Loaded {len(foreign_keys)} foreign key relationships.")

# Example: Display Stored Procedures
for proc in procedures[:5]:
    print(proc['PROCEDURE'])

# Advanced: Linking GIS Data to PACS Database
# Example Linking: Fetch Tax Districts with Associated GIS Data
query = """
SELECT 
    td.TaxDistrictID, 
    td.Name, 
    gis.Geometry 
FROM TaxDistricts td
JOIN GISData gis ON td.TaxDistrictID = gis.TaxDistrictID
"""
linked_data = execute_query(query)
display(linked_data)
```
