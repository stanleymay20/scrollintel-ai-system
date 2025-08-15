#!/usr/bin/env python3
"""Setup script for AI Data Readiness Platform."""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_data_readiness.core.config import Config
from ai_data_readiness.migrations.migration_runner import MigrationRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables and directories."""
    logger.info("Setting up environment...")
    
    # Create necessary directories
    directories = [
        "logs",
        "data",
        "temp",
        "exports",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# AI Data Readiness Platform Configuration

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_data_readiness
DB_USER=postgres
DB_PASSWORD=your_password_here

# Processing Configuration
MAX_WORKERS=4
BATCH_SIZE=1000
MEMORY_LIMIT_GB=8
TEMP_DIR=./temp

# Quality Thresholds
COMPLETENESS_THRESHOLD=0.95
ACCURACY_THRESHOLD=0.90
CONSISTENCY_THRESHOLD=0.85
VALIDITY_THRESHOLD=0.90
AI_READINESS_THRESHOLD=0.80

# Monitoring Configuration
DRIFT_THRESHOLD=0.1
MONITORING_INTERVAL=60
ALERT_EMAIL=admin@yourcompany.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
        env_file.write_text(env_content)
        logger.info("Created .env file with default configuration")
        logger.warning("Please update the .env file with your actual configuration values")


def validate_dependencies():
    """Validate that required dependencies are installed."""
    logger.info("Validating dependencies...")
    
    required_packages = [
        "sqlalchemy",
        "pandas",
        "numpy",
        "scikit-learn",
        "psycopg2"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install requirements: pip install -r ai_data_readiness_requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True


def setup_database():
    """Setup database and run migrations."""
    logger.info("Setting up database...")
    
    try:
        config = Config()
        config.validate()
        
        runner = MigrationRunner(config)
        
        # Check if database setup is needed
        status = runner.check_migration_status()
        
        if not status["database_exists"]:
            logger.info("Database does not exist, creating...")
            if not runner.create_database_if_not_exists():
                logger.error("Failed to create database")
                return False
        
        if status["migration_needed"]:
            logger.info("Running database migrations...")
            if not runner.run_initial_migration():
                logger.error("Failed to run migrations")
                return False
        else:
            logger.info("Database is already up to date")
        
        # Test database connection
        from ai_data_readiness.models.database import Database
        db = Database(config.database.connection_string)
        if not db.health_check():
            logger.error("Database health check failed")
            return False
        
        logger.info("Database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def create_sample_config():
    """Create sample configuration files."""
    logger.info("Creating sample configuration files...")
    
    # Create sample quality config
    quality_config = {
        "quality_dimensions": {
            "completeness": {"weight": 0.25, "threshold": 0.95},
            "accuracy": {"weight": 0.25, "threshold": 0.90},
            "consistency": {"weight": 0.20, "threshold": 0.85},
            "validity": {"weight": 0.20, "threshold": 0.90},
            "uniqueness": {"weight": 0.05, "threshold": 0.95},
            "timeliness": {"weight": 0.05, "threshold": 0.90}
        },
        "ai_readiness_weights": {
            "data_quality": 0.30,
            "feature_quality": 0.25,
            "bias_score": 0.20,
            "compliance_score": 0.15,
            "scalability_score": 0.10
        }
    }
    
    import json
    with open("config/quality_config.json", "w") as f:
        json.dump(quality_config, f, indent=2)
    
    logger.info("Sample configuration files created")


def main():
    """Main setup function."""
    logger.info("Starting AI Data Readiness Platform setup...")
    
    # Create config directory
    Path("config").mkdir(exist_ok=True)
    
    # Setup environment
    setup_environment()
    
    # Validate dependencies
    if not validate_dependencies():
        logger.error("Setup failed due to missing dependencies")
        return False
    
    # Setup database
    if not setup_database():
        logger.error("Setup failed due to database issues")
        return False
    
    # Create sample configurations
    create_sample_config()
    
    logger.info("AI Data Readiness Platform setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Update the .env file with your configuration")
    logger.info("2. Review the configuration files in the config/ directory")
    logger.info("3. Start the platform using: python -m ai_data_readiness.api.main")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)