"""
Seed script to add sample data to the BCBS Values database.

This script creates sample property data, valuations, and agent data
for demonstration purposes.
"""
import random
import uuid
from datetime import datetime, timedelta
from app import app, db
from models import Property, PropertyValuation, PropertyFeature, ETLJob, Agent, AgentLog


def create_sample_properties(count=25):
    """Create sample properties with randomized data."""
    print(f"Creating {count} sample properties...")
    
    # Neighborhoods in Benton County, WA
    neighborhoods = [
        "Meadow Springs", "South Richland", "Horn Rapids", "West Richland",
        "Central Kennewick", "Southridge", "Canyon Lakes", "Columbia Park",
        "Badger Mountain", "Queensgate", "Richland Heights", "Downtown Richland",
        "Clearwater", "Amon Basin", "Rancho Reata", "Finley", "Rancho Reata",
        "West Highlands", "Keene", "Inspiration Estates"
    ]
    
    # Property types
    property_types = ["Single Family", "Townhouse", "Condo", "Multi-Family", "Vacant Land"]
    
    # Street names
    street_names = [
        "Oak", "Maple", "Cedar", "Pine", "Elm", "Spruce", "Birch", "Willow",
        "Cherry", "Sycamore", "Aspen", "Magnolia", "Dogwood", "Juniper",
        "Walnut", "Chestnut", "Hickory", "Beech", "Cypress", "Redwood"
    ]
    
    # Street suffixes
    street_suffixes = ["St", "Ave", "Blvd", "Dr", "Ln", "Rd", "Way", "Pl", "Ct"]
    
    # Create properties
    properties = []
    for i in range(count):
        # Generate a unique property ID
        property_id = f"BC-{random.randint(10000, 99999)}"
        
        # Generate a random address
        street_num = random.randint(100, 9999)
        street_name = random.choice(street_names)
        street_suffix = random.choice(street_suffixes)
        address = f"{street_num} {street_name} {street_suffix}"
        
        # Randomly select city (most are in Richland or Kennewick)
        city_weights = [0.45, 0.45, 0.05, 0.05]  # Richland, Kennewick, West Richland, Prosser
        city = random.choices(
            ["Richland", "Kennewick", "West Richland", "Prosser"],
            weights=city_weights
        )[0]
        
        # Zip codes for Benton County cities
        zip_codes = {
            "Richland": ["99352", "99354"],
            "Kennewick": ["99336", "99337", "99338"],
            "West Richland": ["99353"],
            "Prosser": ["99350"]
        }
        zip_code = random.choice(zip_codes[city])
        
        # Randomize property details
        neighborhood = random.choice(neighborhoods)
        property_type = random.choice(property_types)
        year_built = random.randint(1950, 2023) if property_type != "Vacant Land" else None
        bedrooms = random.randint(2, 5) if property_type not in ["Vacant Land", "Condo"] else (1 if property_type == "Condo" else 0)
        bathrooms = round(random.uniform(1.0, 4.0), 1) if property_type != "Vacant Land" else 0
        square_feet = random.randint(1000, 4000) if property_type != "Vacant Land" else 0
        lot_size = random.randint(3500, 25000)
        
        # Created dates between 1-12 months ago
        days_ago = random.randint(30, 365)
        created_at = datetime.utcnow() - timedelta(days=days_ago)
        
        # Updated dates between creation and now
        update_days_ago = random.randint(0, days_ago - 1)
        updated_at = datetime.utcnow() - timedelta(days=update_days_ago)
        
        # Last sale date (if any)
        if random.random() > 0.3:  # 70% chance to have a last sale date
            sale_days_ago = random.randint(days_ago, days_ago + 1095)  # Up to 3 years before creation
            last_sale_date = datetime.utcnow() - timedelta(days=sale_days_ago)
            last_sale_price = random.randint(250000, 800000)
        else:
            last_sale_date = None
            last_sale_price = None
        
        # Create the property
        property = Property(
            property_id=property_id,
            address=address,
            city=city,
            state="WA",
            zip_code=zip_code,
            neighborhood=neighborhood,
            property_type=property_type,
            year_built=year_built,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            square_feet=square_feet,
            lot_size=lot_size,
            last_sale_date=last_sale_date,
            last_sale_price=last_sale_price,
            created_at=created_at,
            updated_at=updated_at
        )
        
        properties.append(property)
    
    # Save properties to database
    db.session.add_all(properties)
    db.session.commit()
    
    return properties


def create_sample_valuations(properties):
    """Create sample property valuations for given properties."""
    print(f"Creating valuations for {len(properties)} properties...")
    
    # Valuation methods
    methods = [
        "Linear Regression", "Random Forest", "Gradient Boosting", 
        "Ridge Regression", "Elastic Net"
    ]
    
    valuations = []
    for property in properties:
        # Number of valuations for this property (1-3)
        num_valuations = random.randint(1, 3)
        
        for i in range(num_valuations):
            # Valuation date
            if i == 0:
                # Most recent valuation (within last 30 days)
                days_ago = random.randint(0, 30)
            else:
                # Older valuations (30-365 days ago)
                days_ago = random.randint(30 + (i * 30), min(365, 30 + (i * 120)))
            
            valuation_date = datetime.utcnow() - timedelta(days=days_ago)
            
            # Base estimated value
            if property.property_type == "Vacant Land":
                base_value = random.randint(50000, 300000)
            elif property.property_type == "Condo":
                base_value = random.randint(180000, 400000)
            elif property.property_type == "Townhouse":
                base_value = random.randint(250000, 450000)
            elif property.property_type == "Multi-Family":
                base_value = random.randint(400000, 900000)
            else:  # Single Family
                base_value = random.randint(300000, 800000)
            
            # Adjust for property attributes
            if property.square_feet:
                # $150-250 per square foot
                sq_ft_value = property.square_feet * random.uniform(150, 250)
                # Weight the base_value 40% and sq_ft_value 60%
                estimated_value = (base_value * 0.4) + (sq_ft_value * 0.6)
            else:
                estimated_value = base_value
            
            # Add some randomness to make it look realistic
            estimated_value = estimated_value * random.uniform(0.9, 1.1)
            
            # Each valuation is different (increase of 1-5% for newer valuations)
            if i > 0:
                estimated_value = estimated_value * (1 - (random.uniform(0.01, 0.05) * i))
            
            # Round to nearest 100
            estimated_value = round(estimated_value / 100) * 100
            
            # Valuation method
            valuation_method = random.choice(methods)
            
            # Confidence score (0.7 - 0.98)
            confidence_score = random.uniform(0.7, 0.98)
            
            # Sample inputs used for valuation
            inputs = {
                "square_feet": property.square_feet,
                "bedrooms": property.bedrooms,
                "bathrooms": property.bathrooms,
                "year_built": property.year_built,
                "lot_size": property.lot_size,
                "neighborhood_factor": random.uniform(0.9, 1.2),
                "location_score": random.uniform(60, 95),
                "market_trend": random.uniform(-0.02, 0.08)
            }
            
            # Create the valuation
            valuation = PropertyValuation(
                property_id=property.id,
                estimated_value=estimated_value,
                valuation_date=valuation_date,
                valuation_method=valuation_method,
                confidence_score=confidence_score,
                inputs=inputs,
                created_at=valuation_date
            )
            
            valuations.append(valuation)
    
    # Save valuations to database
    db.session.add_all(valuations)
    db.session.commit()
    
    return valuations


def create_sample_property_features(properties):
    """Create sample property features for given properties."""
    print(f"Creating features for {len(properties)} properties...")
    
    features = []
    for property in properties:
        # Skip features for vacant land
        if property.property_type == "Vacant Land":
            continue
        
        # Common features
        feature_list = []
        
        # Add features based on property type
        if property.property_type in ["Single Family", "Townhouse"]:
            # Building features
            if random.random() > 0.5:
                feature_list.append(("Garage Type", random.choice(["Attached", "Detached", "None"])))
            
            if random.random() > 0.3:
                feature_list.append(("Garage Spaces", str(random.randint(1, 3))))
            
            if random.random() > 0.6:
                feature_list.append(("Basement", random.choice(["Finished", "Unfinished", "Partial", "None"])))
            
            # Exterior features
            if random.random() > 0.5:
                feature_list.append(("Exterior Material", random.choice(["Vinyl", "Wood", "Brick", "Stucco", "Fiber Cement"])))
            
            if random.random() > 0.7:
                feature_list.append(("Roof Type", random.choice(["Asphalt Shingle", "Metal", "Tile", "Slate"])))
            
            # Interior features
            if random.random() > 0.4:
                feature_list.append(("Heating Type", random.choice(["Forced Air", "Heat Pump", "Electric", "Gas"])))
            
            if random.random() > 0.6:
                feature_list.append(("Cooling Type", random.choice(["Central Air", "Heat Pump", "None"])))
            
            # Lot features
            if random.random() > 0.5:
                feature_list.append(("Fence", random.choice(["Full", "Partial", "None"])))
            
            if random.random() > 0.7:
                feature_list.append(("Pool", random.choice(["In-ground", "Above-ground", "None"])))
            
            if random.random() > 0.7:
                feature_list.append(("View", random.choice(["Mountain", "Water", "City", "None"])))
        
        elif property.property_type == "Condo":
            # Condo-specific features
            if random.random() > 0.3:
                feature_list.append(("Floor", str(random.randint(1, 20))))
            
            if random.random() > 0.5:
                feature_list.append(("Unit Type", random.choice(["Corner", "Interior", "End"])))
            
            if random.random() > 0.6:
                feature_list.append(("Parking", random.choice(["Assigned", "Garage", "Street", "None"])))
            
            if random.random() > 0.7:
                feature_list.append(("Elevator", random.choice(["Yes", "No"])))
            
            if random.random() > 0.5:
                feature_list.append(("HOA Fee", f"${random.randint(150, 700)}/month"))
        
        elif property.property_type == "Multi-Family":
            # Multi-family specific features
            if random.random() > 0.3:
                feature_list.append(("Units", str(random.randint(2, 8))))
            
            if random.random() > 0.5:
                feature_list.append(("Unit Mix", random.choice(["All 1 BR", "1-2 BR Mix", "Various Sizes"])))
            
            if random.random() > 0.6:
                feature_list.append(("Parking", random.choice(["Assigned", "Open", "Street", "Garage"])))
            
            if random.random() > 0.7:
                feature_list.append(("Laundry", random.choice(["In-unit", "Common Area", "Hookups Only"])))
        
        # Create the features
        for name, value in feature_list:
            feature = PropertyFeature(
                property_id=property.id,
                feature_name=name,
                feature_value=value,
                created_at=property.created_at
            )
            features.append(feature)
    
    # Save features to database
    db.session.add_all(features)
    db.session.commit()
    
    return features


def create_sample_etl_jobs():
    """Create sample ETL job data."""
    print("Creating sample ETL jobs...")
    
    # ETL job types
    job_types = ["Extract", "Transform", "Load", "Full ETL"]
    
    # Data sources
    sources = ["MLS", "NARRPR", "PACS"]
    
    # Job names
    job_names = [
        "MLS Data Import", "NARRPR Property Data Sync", "PACS Assessment Import",
        "Property Geocoding", "School District Assignment", "Neighborhood Mapping"
    ]
    
    # Status options
    statuses = ["completed", "running", "failed"]
    # Weight toward completed (70% completed, 20% running, 10% failed)
    status_weights = [0.7, 0.2, 0.1]
    
    # Create ETL jobs
    jobs = []
    for i in range(20):  # Create 20 sample ETL jobs
        # Job details
        job_name = random.choice(job_names)
        job_type = random.choice(job_types)
        source = random.choice(sources) if random.random() > 0.3 else None
        
        # Timing
        # Most recent jobs in the last 14 days
        days_ago = random.randint(0, 14) 
        start_time = datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(0, 23))
        
        # Status
        status = random.choices(statuses, weights=status_weights)[0]
        
        # For completed or failed jobs, add an end time
        if status != "running":
            # Job took between 5 minutes and 4 hours
            minutes_taken = random.randint(5, 240)
            end_time = start_time + timedelta(minutes=minutes_taken)
        else:
            end_time = None
        
        # For completed jobs, add records processed
        if status == "completed":
            records_processed = random.randint(50, 5000)
            records_failed = random.randint(0, int(records_processed * 0.05))  # Up to 5% failed
        elif status == "running":
            records_processed = random.randint(10, 1000)
            records_failed = random.randint(0, int(records_processed * 0.02))
        else:  # failed
            records_processed = random.randint(0, 500)
            records_failed = random.randint(1, 100)
        
        # Error message for failed jobs
        error_message = None
        if status == "failed":
            error_messages = [
                "Connection timeout to data source",
                "Invalid API response format",
                "Authentication failed",
                "Malformed data in source file",
                "Duplicate entry violation",
                "Memory allocation error",
                "Network connection lost"
            ]
            error_message = random.choice(error_messages)
        
        # Create the ETL job
        job = ETLJob(
            job_name=job_name,
            job_type=job_type,
            source=source,
            start_time=start_time,
            end_time=end_time,
            status=status,
            records_processed=records_processed,
            records_failed=records_failed,
            error_message=error_message
        )
        
        jobs.append(job)
    
    # Save ETL jobs to database
    db.session.add_all(jobs)
    db.session.commit()
    
    return jobs


def create_sample_agents():
    """Create sample agent data."""
    print("Creating sample agents...")
    
    # Agent types
    agent_types = [
        "DataCollector", "PropertyAnalyzer", "ValuationEngine",
        "MarketTrends", "SpatialAnalysis", "ETLManager"
    ]
    
    # Create agent records
    agents = []
    for agent_type in agent_types:
        # Create a unique agent ID
        agent_id = str(uuid.uuid4())
        
        # Generate agent name based on type
        if agent_type == "DataCollector":
            agent_name = "BCBS Data Collector"
        elif agent_type == "PropertyAnalyzer":
            agent_name = "BCBS Property Analyzer"
        elif agent_type == "ValuationEngine":
            agent_name = "BCBS Valuation Engine"
        elif agent_type == "MarketTrends":
            agent_name = "BCBS Market Trend Agent"
        elif agent_type == "SpatialAnalysis":
            agent_name = "BCBS GIS Analysis Agent"
        else:
            agent_name = "BCBS ETL Manager"
        
        # Randomize status (biased toward idle)
        status_choices = ["idle", "running", "error"]
        status_weights = [0.6, 0.3, 0.1]
        status = random.choices(status_choices, weights=status_weights)[0]
        
        # Last heartbeat
        if status == "idle" or status == "running":
            # Recent heartbeat for active agents
            minutes_ago = random.randint(0, 30)
            last_heartbeat = datetime.utcnow() - timedelta(minutes=minutes_ago)
        else:
            # Older heartbeat for error agents
            hours_ago = random.randint(2, 48)
            last_heartbeat = datetime.utcnow() - timedelta(hours=hours_ago)
        
        # Current task for running agents
        current_task = None
        if status == "running":
            tasks = [
                "Fetching MLS data", "Processing property records",
                "Analyzing market trends", "Running valuation models",
                "Updating spatial data", "Assigning neighborhood scores"
            ]
            current_task = random.choice(tasks)
        
        # Queue size
        queue_size = 0
        if status == "running":
            queue_size = random.randint(1, 10)
        elif status == "idle":
            # Occasionally idle agents have a queue
            if random.random() > 0.7:
                queue_size = random.randint(1, 3)
        
        # Success rate
        if status == "error":
            success_rate = random.uniform(70.0, 90.0)
        else:
            success_rate = random.uniform(95.0, 100.0)
        
        # Error count
        if status == "error":
            error_count = random.randint(3, 20)
        else:
            error_count = random.randint(0, 2)
        
        # Created date
        days_ago = random.randint(30, 365)
        created_at = datetime.utcnow() - timedelta(days=days_ago)
        
        # Updated date
        if status == "running" or status == "idle":
            updated_at = last_heartbeat
        else:
            updated_at = datetime.utcnow() - timedelta(hours=random.randint(2, 48))
        
        # Create the agent
        agent = Agent(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            status=status,
            last_heartbeat=last_heartbeat,
            current_task=current_task,
            queue_size=queue_size,
            success_rate=success_rate,
            error_count=error_count,
            created_at=created_at,
            updated_at=updated_at
        )
        
        agents.append(agent)
    
    # Save agents to database
    db.session.add_all(agents)
    db.session.commit()
    
    return agents


def create_sample_agent_logs(agents):
    """Create sample agent logs."""
    print(f"Creating logs for {len(agents)} agents...")
    
    logs = []
    for agent in agents:
        # Number of logs for this agent (5-20)
        num_logs = random.randint(5, 20)
        
        for i in range(num_logs):
            # Log timestamp
            hours_ago = random.randint(0, 24 * 14)  # Up to 14 days ago
            timestamp = datetime.utcnow() - timedelta(hours=hours_ago)
            
            # Log level
            levels = ["info", "warning", "error", "success"]
            level_weights = [0.7, 0.1, 0.05, 0.15]  # Mostly info logs
            level = random.choices(levels, weights=level_weights)[0]
            
            # Log message based on agent type and level
            if level == "info":
                if agent.agent_type == "DataCollector":
                    messages = [
                        "Starting data collection from MLS",
                        "Processing property batch",
                        "Downloaded 214 new property records",
                        "Connecting to NARRPR API",
                        "Synchronizing property data"
                    ]
                elif agent.agent_type == "PropertyAnalyzer":
                    messages = [
                        "Analyzing property features",
                        "Computing comparables for property BC-34567",
                        "Updating property metrics",
                        "Classifying property types",
                        "Generating property summaries"
                    ]
                elif agent.agent_type == "ValuationEngine":
                    messages = [
                        "Starting valuation cycle",
                        "Training model with latest data",
                        "Loaded 5 valuation models",
                        "Executing batch valuations",
                        "Model performance: RMSE = 24563.21"
                    ]
                else:
                    messages = [
                        "Agent initialized",
                        "Checking for new tasks",
                        "Processing queue items",
                        "Idle, waiting for tasks",
                        "Heartbeat sent"
                    ]
            elif level == "warning":
                messages = [
                    "API rate limit approaching",
                    "Slow response from data source",
                    "Missing data for property BC-45678",
                    "Model convergence taking longer than expected",
                    "High memory usage detected"
                ]
            elif level == "error":
                messages = [
                    "Connection failed after 3 retries",
                    "API authentication error",
                    "Database query timeout",
                    "Failed to process property record",
                    "Model training failed: insufficient data"
                ]
            else:  # success
                messages = [
                    "Successfully processed 125 properties",
                    "Valuation batch completed",
                    "Model training finished with 0.92 accuracy",
                    "ETL process completed successfully",
                    "All tasks in queue completed"
                ]
            
            message = random.choice(messages)
            
            # Create the log
            log = AgentLog(
                agent_id=agent.id,
                level=level,
                message=message,
                timestamp=timestamp
            )
            
            logs.append(log)
    
    # Save logs to database
    db.session.add_all(logs)
    db.session.commit()
    
    return logs


def main():
    """Main function to create all sample data."""
    with app.app_context():
        # Check if we already have data
        existing_properties = Property.query.count()
        if existing_properties > 0:
            print(f"Database already contains {existing_properties} properties.")
            confirm = input("Do you want to proceed and add more sample data? (y/n): ")
            if confirm.lower() != 'y':
                print("Exiting without changes.")
                return
        
        # Create the sample data
        properties = create_sample_properties(25)
        create_sample_valuations(properties)
        create_sample_property_features(properties)
        create_sample_etl_jobs()
        agents = create_sample_agents()
        create_sample_agent_logs(agents)
        
        print("Sample data creation complete!")


if __name__ == "__main__":
    main()