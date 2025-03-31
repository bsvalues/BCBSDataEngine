"""
Seed script to populate the database with initial data.
This includes sample users, properties, and valuations.
"""
import os
import logging
import random
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
from app import app, db
from models import User, Property, PropertyValuation, ApiKey, EtlStatus, DataSource, Agent, AgentLog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data
NEIGHBORHOODS = [
    "Benton County", "Downtown", "Westside", "Northview", "Eastwood",
    "Southside", "Riverside", "Mountain View", "Lakeview", "Parkside"
]

PROPERTY_TYPES = ["single_family", "condo", "townhouse", "multi_family", "land"]

VALUATION_METHODS = [
    "enhanced_regression", "lightgbm", "xgboost",
    "linear_regression", "ridge_regression", "lasso_regression", "elastic_net"
]

AGENT_TYPES = ["regression", "ensemble", "gis"]

STATUSES = ["idle", "active", "error", "completed", "processing", "queued"]

# Functions to create sample data
def create_sample_users():
    """Create sample users if none exist."""
    if User.query.count() == 0:
        logger.info("Creating sample users...")
        
        # Admin user
        admin = User(
            username="admin",
            email="admin@example.com",
            password_hash=generate_password_hash("Password123!")
        )
        
        # Demo user
        demo = User(
            username="demo",
            email="demo@example.com",
            password_hash=generate_password_hash("Demo123!")
        )
        
        # Test user
        test = User(
            username="test",
            email="test@example.com",
            password_hash=generate_password_hash("Test123!")
        )
        
        db.session.add_all([admin, demo, test])
        db.session.commit()
        
        # Create API keys for admin
        api_key = ApiKey(
            key="api-key-12345-admin",
            name="Admin API Key",
            user_id=admin.id,
            is_active=True
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        logger.info(f"Created {User.query.count()} sample users")
    else:
        logger.info("Users already exist, skipping creation")

def create_sample_properties():
    """Create sample properties if fewer than 10 exist."""
    if Property.query.count() < 10:
        logger.info("Creating sample properties...")
        
        # Generate random properties
        properties = []
        for i in range(20):
            # Determine property type
            property_type = random.choice(PROPERTY_TYPES)
            
            # Generate appropriate values based on property type
            if property_type == "single_family":
                bedrooms = random.randint(2, 5)
                bathrooms = random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
                living_area = random.randint(1200, 3500)
                land_area = random.uniform(0.1, 1.0)
            elif property_type == "condo":
                bedrooms = random.randint(1, 3)
                bathrooms = random.choice([1.0, 1.5, 2.0, 2.5])
                living_area = random.randint(700, 1800)
                land_area = None
            elif property_type == "townhouse":
                bedrooms = random.randint(2, 4)
                bathrooms = random.choice([1.5, 2.0, 2.5, 3.0])
                living_area = random.randint(1000, 2200)
                land_area = random.uniform(0.05, 0.25)
            elif property_type == "multi_family":
                bedrooms = random.randint(4, 8)
                bathrooms = random.choice([2.0, 2.5, 3.0, 3.5, 4.0])
                living_area = random.randint(2000, 4500)
                land_area = random.uniform(0.2, 1.5)
            else:  # land
                bedrooms = None
                bathrooms = None
                living_area = None
                land_area = random.uniform(0.5, 10.0)
            
            # Create property object
            property_obj = Property(
                property_id=f"PROP-{1000 + i}",
                address=f"{100 + i} {random.choice(['Main', 'Oak', 'Maple', 'Pine', 'Cedar'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr', 'Ln'])}",
                city="Sample City",
                state="WA",
                zip_code=f"9801{random.randint(0, 9)}",
                neighborhood=random.choice(NEIGHBORHOODS),
                property_type=property_type,
                year_built=random.randint(1950, 2023) if property_type != "land" else None,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                living_area=living_area,
                land_area=land_area
            )
            
            properties.append(property_obj)
        
        db.session.add_all(properties)
        db.session.commit()
        
        logger.info(f"Created {len(properties)} sample properties")
    else:
        logger.info("Sufficient properties already exist, skipping creation")

def create_sample_valuations():
    """Create sample property valuations."""
    properties = Property.query.all()
    if PropertyValuation.query.count() < len(properties):
        logger.info("Creating sample property valuations...")
        
        valuations = []
        for property_obj in properties:
            # Base value calculation
            if property_obj.property_type == "land":
                # Land is valued by acre
                base_value = 100000 + (property_obj.land_area * 250000)
            else:
                # Calculate base value from square footage
                sq_ft_value = {
                    "single_family": random.uniform(250, 350),
                    "condo": random.uniform(300, 400),
                    "townhouse": random.uniform(275, 375),
                    "multi_family": random.uniform(200, 300)
                }.get(property_obj.property_type, 250)
                
                base_value = property_obj.living_area * sq_ft_value
                
                # Adjust for bedrooms and bathrooms
                if property_obj.bedrooms:
                    base_value += property_obj.bedrooms * 15000
                
                if property_obj.bathrooms:
                    base_value += property_obj.bathrooms * 25000
                
                # Adjust for age if year_built is available
                if property_obj.year_built:
                    age = 2023 - property_obj.year_built
                    age_factor = max(0.7, 1 - (age / 100))
                    base_value *= age_factor
            
            # Create 1-3 valuations for each property with slightly different values
            for i in range(random.randint(1, 3)):
                # Add some randomness to the value
                adjusted_value = base_value * random.uniform(0.9, 1.1)
                
                # Select a valuation method with different probabilities
                method_weights = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]  # Weights for each method
                valuation_method = random.choices(VALUATION_METHODS, weights=method_weights, k=1)[0]
                
                # Determine confidence score based on method
                if valuation_method in ["enhanced_regression", "lightgbm", "xgboost"]:
                    confidence = random.uniform(0.8, 0.95)  # Higher confidence for advanced methods
                else:
                    confidence = random.uniform(0.6, 0.85)  # Lower confidence for basic methods
                
                # Randomize valuation date over the past year
                days_ago = random.randint(0, 365)
                valuation_date = datetime.utcnow() - timedelta(days=days_ago)
                
                valuation = PropertyValuation(
                    property_id=property_obj.id,
                    estimated_value=round(adjusted_value, 2),
                    confidence_score=confidence,
                    valuation_date=valuation_date,
                    valuation_method=valuation_method
                )
                
                valuations.append(valuation)
        
        db.session.add_all(valuations)
        db.session.commit()
        
        logger.info(f"Created {len(valuations)} sample valuations")
    else:
        logger.info("Sufficient valuations already exist, skipping creation")

def create_sample_agents():
    """Create sample agents if none exist."""
    if Agent.query.count() == 0:
        logger.info("Creating sample agents...")
        
        agents = []
        # Create several agents of each type
        for agent_type in AGENT_TYPES:
            for i in range(3):  # 3 of each type
                # Determine status with probabilities
                status_weights = [0.6, 0.3, 0.1]  # idle, active, error
                status = random.choices(["idle", "active", "error"], weights=status_weights, k=1)[0]
                
                # Set queue size based on status
                if status == "active":
                    queue_size = random.randint(1, 20)
                    total_processed = random.randint(100, 1000)
                    success_rate = random.uniform(0.85, 0.99)
                    avg_time = random.uniform(0.5, 3.0)
                    last_active = datetime.utcnow() - timedelta(minutes=random.randint(1, 30))
                elif status == "idle":
                    queue_size = 0
                    total_processed = random.randint(50, 500)
                    success_rate = random.uniform(0.8, 0.98)
                    avg_time = random.uniform(0.7, 3.5)
                    last_active = datetime.utcnow() - timedelta(hours=random.randint(1, 24))
                else:  # error
                    queue_size = random.randint(0, 10)
                    total_processed = random.randint(20, 300)
                    success_rate = random.uniform(0.5, 0.85)
                    avg_time = random.uniform(1.0, 5.0)
                    last_active = datetime.utcnow() - timedelta(hours=random.randint(1, 48))
                
                agent = Agent(
                    agent_id=f"agent-{agent_type}-{i+1:02d}",
                    agent_type=agent_type,
                    status=status,
                    queue_size=queue_size,
                    total_processed=total_processed,
                    success_rate=success_rate,
                    average_processing_time=avg_time,
                    last_active=last_active
                )
                
                agents.append(agent)
        
        db.session.add_all(agents)
        db.session.commit()
        
        # Create sample agent logs
        logs = []
        for agent in agents:
            log_count = random.randint(5, 15)
            for j in range(log_count):
                # Determine log level with probabilities
                if agent.status == "error":
                    level_weights = [0.2, 0.3, 0.5]  # info, warning, error
                else:
                    level_weights = [0.7, 0.2, 0.1]  # info, warning, error
                
                level = random.choices(["info", "warning", "error"], weights=level_weights, k=1)[0]
                
                # Generate log message based on level
                if level == "info":
                    messages = [
                        "Processing request completed successfully",
                        "Agent started and initialized correctly",
                        "Connection established with data source",
                        "Processed batch of 10 properties",
                        "Updated model parameters"
                    ]
                elif level == "warning":
                    messages = [
                        "Slow response time detected",
                        "Missing optional parameters in request",
                        "Falling back to secondary data source",
                        "Rate limit approaching threshold",
                        "Resource usage nearing capacity"
                    ]
                else:  # error
                    messages = [
                        "Failed to connect to database",
                        "Invalid input parameters received",
                        "Out of memory error during processing",
                        "Model prediction failed",
                        "Timeout waiting for external service"
                    ]
                
                message = random.choice(messages)
                timestamp = agent.last_active - timedelta(minutes=random.randint(1, 1440))
                
                log = AgentLog(
                    agent_id=agent.id,
                    level=level,
                    message=message,
                    timestamp=timestamp
                )
                
                logs.append(log)
        
        db.session.add_all(logs)
        db.session.commit()
        
        logger.info(f"Created {len(agents)} sample agents with {len(logs)} log entries")
    else:
        logger.info("Agents already exist, skipping creation")

def create_sample_etl_status():
    """Create sample ETL status records if none exist."""
    if EtlStatus.query.count() == 0:
        logger.info("Creating sample ETL status records...")
        
        etl_status = EtlStatus(
            status="completed",
            progress=1.0,
            records_processed=random.randint(1000, 5000),
            success_rate=random.uniform(0.9, 0.99),
            average_processing_time=random.uniform(0.1, 0.5),
            completeness=random.uniform(0.85, 0.98),
            accuracy=random.uniform(0.9, 0.98),
            timeliness=random.uniform(0.8, 0.95),
            last_update=datetime.utcnow() - timedelta(hours=random.randint(1, 48))
        )
        
        db.session.add(etl_status)
        db.session.commit()
        
        # Create associated data sources
        sources = []
        source_names = ["county_records", "mls_listings", "tax_assessments", "census_data", "sales_transactions"]
        for name in source_names:
            # Determine status
            status_weights = [0.1, 0.1, 0.1, 0.6, 0.1]  # idle, queued, processing, completed, error
            status = random.choices(STATUSES[:-1], weights=status_weights, k=1)[0]
            
            source = DataSource(
                name=name,
                status=status,
                records=random.randint(100, 1000),
                etl_status_id=etl_status.id,
                created_at=etl_status.last_update - timedelta(hours=random.randint(1, 24)),
                updated_at=etl_status.last_update
            )
            
            sources.append(source)
        
        db.session.add_all(sources)
        db.session.commit()
        
        logger.info(f"Created ETL status with {len(sources)} data sources")
    else:
        logger.info("ETL status records already exist, skipping creation")

def seed_database():
    """Seed the database with sample data."""
    with app.app_context():
        create_sample_users()
        create_sample_properties()
        create_sample_valuations()
        create_sample_agents()
        create_sample_etl_status()

if __name__ == "__main__":
    seed_database()
    logger.info("Database seeding completed successfully")