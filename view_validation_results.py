import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

def view_validation_results():
    # Create database connection
    engine = create_engine(os.environ.get('DATABASE_URL'))
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Query validation results
    validation_results = pd.read_sql('SELECT * FROM validation_results ORDER BY timestamp DESC LIMIT 3', engine)
    
    print(f'Recent validation results: {len(validation_results)}')
    
    for i, row in validation_results.iterrows():
        print(f"\nValidation {i+1}:")
        print(f"Timestamp: {row['timestamp']}")
        print(f"Status: {row['status']}")
        print(f"Results (partial): {row['results'][:150]}...")

if __name__ == "__main__":
    view_validation_results()