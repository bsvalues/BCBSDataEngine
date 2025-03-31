"""
Seed script for BCBS Values application.

This script populates the database with sample data for development and testing purposes.
"""
import os
import random
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

from app import app, db
from models import (
    User, Property, Valuation, Agent, AgentLog, ETLJob, ApiKey, PropertyImage, MarketTrend
)


def create_users():
    """Create sample users"""
    print("Creating sample users...")
    
    # Admin user
    admin = User(
        username="admin",
        email="admin@bcbsvalues.com",
        password_hash=generate_password_hash("adminpassword"),
        first_name="Admin",
        last_name="User",
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
        is_active=True,
        is_admin=True
    )
    
    # Regular user
    user = User(
        username="user",
        email="user@bcbsvalues.com",
        password_hash=generate_password_hash("userpassword"),
        first_name="Demo",
        last_name="User",
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
        is_active=True,
        is_admin=False
    )
    
    db.session.add(admin)
    db.session.add(user)
    db.session.commit()
    
    print(f"Created {User.query.count()} users")
    return admin, user


def create_properties(owner):
    """Create sample properties"""
    print("Creating sample properties...")
    
    # Sample property data
    sample_properties = [
        {
            "address": "123 Main St",
            "city": "Seattle",
            "state": "WA",
            "zip_code": "98101",
            "property_type": "residential",
            "bedrooms": 3,
            "bathrooms": 2.5,
            "square_feet": 2200,
            "lot_size": 0.25,
            "year_built": 1998,
            "last_sold_date": datetime(2020, 5, 15),
            "last_sold_price": 750000,
            "latitude": 47.6062,
            "longitude": -122.3321,
            "neighborhood": "Downtown",
            "description": "Beautiful modern home in downtown Seattle with stunning views of the Space Needle.",
            "features": "Hardwood floors, granite countertops, stainless steel appliances, deck, garden"
        },
        {
            "address": "456 Oak Ave",
            "city": "Seattle",
            "state": "WA",
            "zip_code": "98115",
            "property_type": "residential",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2800,
            "lot_size": 0.3,
            "year_built": 2005,
            "last_sold_date": datetime(2021, 3, 10),
            "last_sold_price": 950000,
            "latitude": 47.6792,
            "longitude": -122.3860,
            "neighborhood": "Ravenna",
            "description": "Spacious family home in the desirable Ravenna neighborhood, close to parks and schools.",
            "features": "Open floor plan, chef's kitchen, master suite, finished basement, large backyard"
        },
        {
            "address": "789 Elm St",
            "city": "Bellevue",
            "state": "WA",
            "zip_code": "98004",
            "property_type": "residential",
            "bedrooms": 5,
            "bathrooms": 4.5,
            "square_feet": 3500,
            "lot_size": 0.4,
            "year_built": 2012,
            "last_sold_date": datetime(2022, 1, 20),
            "last_sold_price": 1250000,
            "latitude": 47.6101,
            "longitude": -122.2015,
            "neighborhood": "Downtown Bellevue",
            "description": "Luxury home in the heart of Bellevue with modern design and high-end finishes.",
            "features": "Smart home technology, home theater, wine cellar, 3-car garage, rooftop terrace"
        },
        {
            "address": "101 Pine Rd",
            "city": "Kirkland",
            "state": "WA",
            "zip_code": "98033",
            "property_type": "residential",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 1800,
            "lot_size": 0.2,
            "year_built": 1985,
            "last_sold_date": datetime(2019, 9, 5),
            "last_sold_price": 650000,
            "latitude": 47.6769,
            "longitude": -122.2060,
            "neighborhood": "Juanita",
            "description": "Charming rambler in Juanita with recent updates and a beautiful garden.",
            "features": "Updated kitchen, new roof, fresh paint, fenced yard, deck"
        },
        {
            "address": "202 Cedar Blvd",
            "city": "Benton City",
            "state": "WA",
            "zip_code": "99320",
            "property_type": "residential",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2500,
            "lot_size": 0.5,
            "year_built": 2000,
            "last_sold_date": datetime(2018, 7, 12),
            "last_sold_price": 400000,
            "latitude": 46.2630,
            "longitude": -119.4872,
            "neighborhood": "Red Mountain",
            "description": "Spacious home with stunning views of Red Mountain vineyards in Benton City.",
            "features": "Vaulted ceilings, gourmet kitchen, covered patio, wine room, 3-car garage"
        },
        {
            "address": "303 Maple Dr",
            "city": "Richland",
            "state": "WA",
            "zip_code": "99352",
            "property_type": "residential",
            "bedrooms": 3,
            "bathrooms": 2.5,
            "square_feet": 2200,
            "lot_size": 0.3,
            "year_built": 1995,
            "last_sold_date": datetime(2020, 2, 28),
            "last_sold_price": 380000,
            "latitude": 46.2853,
            "longitude": -119.2924,
            "neighborhood": "Horn Rapids",
            "description": "Well-maintained home in Horn Rapids with easy access to outdoor recreation.",
            "features": "Open concept, granite counters, gas fireplace, covered patio, sprinkler system"
        },
        {
            "address": "404 Birch Ln",
            "city": "Kennewick",
            "state": "WA",
            "zip_code": "99336",
            "property_type": "residential",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2800,
            "lot_size": 0.25,
            "year_built": 2010,
            "last_sold_date": datetime(2021, 5, 15),
            "last_sold_price": 450000,
            "latitude": 46.2087,
            "longitude": -119.1651,
            "neighborhood": "Southridge",
            "description": "Modern family home in the Southridge area with mountain views.",
            "features": "Bonus room, large kitchen island, walk-in closets, covered deck, landscaped yard"
        },
        {
            "address": "505 Fir Ave",
            "city": "Pasco",
            "state": "WA",
            "zip_code": "99301",
            "property_type": "residential",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 1900,
            "lot_size": 0.2,
            "year_built": 2005,
            "last_sold_date": datetime(2019, 11, 10),
            "last_sold_price": 320000,
            "latitude": 46.2395,
            "longitude": -119.1005,
            "neighborhood": "Road 68",
            "description": "Lovely single-level home in the popular Road 68 area of Pasco.",
            "features": "Tile floors, stainless appliances, fenced yard, garden beds, RV parking"
        },
        {
            "address": "600 Commercial St",
            "city": "Seattle",
            "state": "WA",
            "zip_code": "98109",
            "property_type": "commercial",
            "bedrooms": None,
            "bathrooms": 4,
            "square_feet": 5000,
            "lot_size": 0.3,
            "year_built": 2001,
            "last_sold_date": datetime(2018, 4, 20),
            "last_sold_price": 1800000,
            "latitude": 47.6205,
            "longitude": -122.3447,
            "neighborhood": "South Lake Union",
            "description": "Prime commercial space in South Lake Union with excellent visibility and foot traffic.",
            "features": "Open floor plan, large windows, modern HVAC, secure entry, parking"
        },
        {
            "address": "700 Industrial Pkwy",
            "city": "Kennewick",
            "state": "WA",
            "zip_code": "99337",
            "property_type": "industrial",
            "bedrooms": None,
            "bathrooms": 2,
            "square_feet": 12000,
            "lot_size": 2.5,
            "year_built": 1995,
            "last_sold_date": datetime(2017, 8, 15),
            "last_sold_price": 1200000,
            "latitude": 46.1841,
            "longitude": -119.1371,
            "neighborhood": "Industrial District",
            "description": "Versatile industrial building with warehouse and office space in Kennewick.",
            "features": "Loading docks, high ceilings, 3-phase power, climate controlled office, fenced yard"
        }
    ]
    
    properties = []
    for prop_data in sample_properties:
        property = Property(
            address=prop_data["address"],
            city=prop_data["city"],
            state=prop_data["state"],
            zip_code=prop_data["zip_code"],
            property_type=prop_data["property_type"],
            bedrooms=prop_data["bedrooms"],
            bathrooms=prop_data["bathrooms"],
            square_feet=prop_data["square_feet"],
            lot_size=prop_data["lot_size"],
            year_built=prop_data["year_built"],
            last_sold_date=prop_data["last_sold_date"],
            last_sold_price=prop_data["last_sold_price"],
            latitude=prop_data["latitude"],
            longitude=prop_data["longitude"],
            neighborhood=prop_data["neighborhood"],
            description=prop_data["description"],
            features=prop_data["features"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            owner_id=owner.id
        )
        db.session.add(property)
        properties.append(property)
    
    db.session.commit()
    print(f"Created {len(properties)} properties")
    return properties


def create_agents():
    """Create sample valuation agents"""
    print("Creating sample agents...")
    
    agent_types = ['regression', 'lightgbm', 'xgboost', 'ensemble', 'gis']
    statuses = ['idle', 'processing', 'idle', 'idle', 'error']
    
    agents = []
    for i, agent_type in enumerate(agent_types):
        agent = Agent(
            name=f"{agent_type.capitalize()} Agent",
            agent_type=agent_type,
            description=f"Agent for {agent_type} valuation method",
            status=statuses[i],
            model_version=f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
            created_at=datetime.utcnow() - timedelta(days=random.randint(10, 100)),
            last_active=datetime.utcnow() - timedelta(minutes=random.randint(5, 120)),
            processing_count=random.randint(100, 1000),
            success_count=random.randint(90, 950),
            error_count=random.randint(0, 50),
            average_confidence=random.uniform(0.7, 0.95),
            configuration=json.dumps({
                "parameters": {
                    "learning_rate": 0.01,
                    "max_depth": 6,
                    "n_estimators": 100
                }
            })
        )
        db.session.add(agent)
        agents.append(agent)
    
    db.session.commit()
    
    # Create some agent logs
    for agent in agents:
        log_levels = ['info', 'warning', 'error', 'debug']
        for _ in range(random.randint(3, 10)):
            log_level = random.choice(log_levels)
            log = AgentLog(
                agent_id=agent.id,
                log_level=log_level,
                message=f"Sample {log_level} log message for {agent.name}",
                timestamp=datetime.utcnow() - timedelta(minutes=random.randint(5, 120)),
                details=json.dumps({
                    "request_id": f"req-{random.randint(1000, 9999)}",
                    "property_id": random.randint(1, 10),
                    "execution_time": f"{random.uniform(0.1, 5.0):.2f} seconds"
                })
            )
            db.session.add(log)
    
    db.session.commit()
    print(f"Created {len(agents)} agents with logs")
    return agents


def create_valuations(properties, user, agents):
    """Create sample property valuations"""
    print("Creating sample valuations...")
    
    valuations = []
    for property in properties:
        # Create 1-3 valuations for each property
        for _ in range(random.randint(1, 3)):
            agent = random.choice(agents)
            
            # Base value around the last sold price with some variation
            base_value = property.last_sold_price if property.last_sold_price else 500000
            estimated_value = base_value * random.uniform(0.9, 1.2)
            
            valuation = Valuation(
                property_id=property.id,
                user_id=user.id,
                agent_id=agent.id,
                estimated_value=estimated_value,
                confidence_score=random.uniform(0.6, 0.98),
                valuation_date=datetime.utcnow() - timedelta(days=random.randint(0, 90)),
                valuation_method=agent.agent_type,
                model_version=agent.model_version,
                adjusted_value=estimated_value * random.uniform(0.95, 1.05),
                adjustment_factors=json.dumps({
                    "location": random.uniform(-0.05, 0.1),
                    "condition": random.uniform(-0.03, 0.05),
                    "market_trend": random.uniform(-0.02, 0.08),
                    "seasonal": random.uniform(-0.01, 0.02)
                }),
                comparable_properties=json.dumps([
                    {"id": random.randint(1, 10), "address": f"{random.randint(100, 999)} Sample St", "similarity": random.uniform(0.7, 0.95)},
                    {"id": random.randint(1, 10), "address": f"{random.randint(100, 999)} Example Ave", "similarity": random.uniform(0.7, 0.95)},
                    {"id": random.randint(1, 10), "address": f"{random.randint(100, 999)} Test Blvd", "similarity": random.uniform(0.7, 0.95)}
                ]),
                metrics=json.dumps({
                    "rmse": random.uniform(10000, 50000),
                    "mae": random.uniform(8000, 35000),
                    "r2": random.uniform(0.65, 0.9)
                }),
                notes=f"Sample valuation created by {agent.name}"
            )
            db.session.add(valuation)
            valuations.append(valuation)
    
    db.session.commit()
    print(f"Created {len(valuations)} valuations")
    return valuations


def create_market_trends():
    """Create sample market trend data"""
    print("Creating sample market trends...")
    
    # Sample neighborhoods
    neighborhoods = [
        {"name": "Downtown", "city": "Seattle", "state": "WA"},
        {"name": "Ravenna", "city": "Seattle", "state": "WA"},
        {"name": "Downtown Bellevue", "city": "Bellevue", "state": "WA"},
        {"name": "Juanita", "city": "Kirkland", "state": "WA"},
        {"name": "Red Mountain", "city": "Benton City", "state": "WA"},
        {"name": "Horn Rapids", "city": "Richland", "state": "WA"},
        {"name": "Southridge", "city": "Kennewick", "state": "WA"},
        {"name": "Road 68", "city": "Pasco", "state": "WA"}
    ]
    
    # Create trends for the last 6 months
    trends = []
    for month in range(6):
        trend_date = datetime.now().replace(day=1) - timedelta(days=30 * month)
        
        for neighborhood in neighborhoods:
            trend = MarketTrend(
                neighborhood=neighborhood["name"],
                city=neighborhood["city"],
                state=neighborhood["state"],
                trend_date=trend_date.date(),
                median_price=random.randint(350000, 1200000),
                average_price=random.randint(400000, 1300000),
                price_per_sqft=random.randint(300, 800),
                inventory_count=random.randint(5, 50),
                days_on_market=random.randint(10, 60),
                month_over_month=random.uniform(-0.03, 0.05),
                year_over_year=random.uniform(0.02, 0.15),
                property_type="residential"
            )
            db.session.add(trend)
            trends.append(trend)
    
    db.session.commit()
    print(f"Created {len(trends)} market trends")
    return trends


def create_etl_jobs():
    """Create sample ETL job records"""
    print("Creating sample ETL jobs...")
    
    job_types = ['import', 'transform', 'export', 'validate']
    sources = ['mls', 'tax_records', 'census_data', 'zillow', 'redfin']
    statuses = ['completed', 'running', 'failed', 'pending']
    
    jobs = []
    for i in range(10):
        job_type = random.choice(job_types)
        status = random.choice(statuses)
        start_time = datetime.utcnow() - timedelta(days=random.randint(0, 30))
        
        total_records = random.randint(1000, 10000) if status != 'pending' else None
        records_processed = random.randint(0, total_records) if total_records and status == 'running' else \
                           total_records if status == 'completed' else \
                           random.randint(0, total_records // 2) if status == 'failed' else 0
        
        end_time = start_time + timedelta(minutes=random.randint(10, 120)) if status in ['completed', 'failed'] else None
        
        job = ETLJob(
            job_type=job_type,
            source=random.choice(sources),
            status=status,
            start_time=start_time,
            end_time=end_time,
            records_processed=records_processed,
            total_records=total_records,
            success_count=records_processed if status == 'completed' else random.randint(0, records_processed) if records_processed else 0,
            error_count=0 if status == 'completed' else random.randint(1, 100) if status == 'failed' else random.randint(0, 10),
            error_details=json.dumps({"error": "Sample error message", "location": "sample_file.py:123"}) if status == 'failed' else None,
            configuration=json.dumps({
                "batch_size": 100,
                "timeout": 3600,
                "retry_count": 3
            })
        )
        db.session.add(job)
        jobs.append(job)
    
    db.session.commit()
    print(f"Created {len(jobs)} ETL jobs")
    return jobs


def create_api_keys(user):
    """Create sample API keys"""
    print("Creating sample API keys...")
    
    # Create one active API key for the user
    api_key = ApiKey(
        key="sample_api_key_for_testing_only",
        name="Test API Key",
        user_id=user.id,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=365),
        last_used=datetime.utcnow() - timedelta(days=1),
        is_active=True,
        permissions="read"
    )
    
    db.session.add(api_key)
    db.session.commit()
    print(f"Created sample API key")
    return api_key


def seed_database():
    """Main function to seed the database with sample data"""
    print("Starting database seeding...")
    
    # Check if database already has data
    if User.query.count() > 0:
        print("Database already has data. Skipping seeding.")
        return
    
    # Create sample data
    admin, user = create_users()
    properties = create_properties(user)
    agents = create_agents()
    valuations = create_valuations(properties, user, agents)
    market_trends = create_market_trends()
    etl_jobs = create_etl_jobs()
    api_key = create_api_keys(user)
    
    print("Database seeding completed successfully!")


if __name__ == "__main__":
    with app.app_context():
        seed_database()