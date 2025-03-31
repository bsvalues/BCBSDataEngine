import os
import sys

# Print Python version and paths
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Try to import required packages
try:
    import flask
    print(f"Flask version: {flask.__version__}")
except ImportError:
    print("Flask is not installed")

try:
    import flask_sqlalchemy
    print(f"Flask-SQLAlchemy version: {flask_sqlalchemy.__version__}")
except ImportError:
    print("Flask-SQLAlchemy is not installed")

try:
    import sqlalchemy
    print(f"SQLAlchemy version: {sqlalchemy.__version__}")
except ImportError:
    print("SQLAlchemy is not installed")

try:
    import flask_wtf
    print(f"Flask-WTF version: {flask_wtf.__version__}")
except ImportError:
    print("Flask-WTF is not installed")

try:
    from flask_login import LoginManager
    print("Flask-Login is installed")
except ImportError:
    print("Flask-Login is not installed")

try:
    import psycopg2
    print(f"psycopg2 version: {psycopg2.__version__}")
except ImportError:
    print("psycopg2 is not installed")

try:
    import dotenv
    print("python-dotenv is installed")
except ImportError:
    print("python-dotenv is not installed")

# Database URL
db_url = os.environ.get("DATABASE_URL")
print(f"DATABASE_URL: {'Present' if db_url else 'Not present'}")
