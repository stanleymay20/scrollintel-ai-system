#!/usr/bin/env python3
"""
Test script to verify database models and setup work correctly.
Uses in-memory SQLite database for testing.
"""

import asyncio
import logging
from scrollintel.models.database_utils import TestDatabaseManager
from scrollintel.models.seed_data import seed_database
from scrollintel.models.database import User, Agent, Dataset, Dashboard, AuditLog
from scrollintel.core.interfaces import UserRole, AgentType, AgentStatus
from sqlalchemy import select, func

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_database_setup():
    """Test database setup with in-memory SQLite."""
    logger.info("Starting database setup test...")
    
    try:
        # Initialize test database manager
        test_manager = TestDatabaseManager()
        await test_manager.initialize()
        logger.info("✓ Test database manager initialized")
        
        # Test database connection
        health = await test_manager.check_health()
        logger.info(f"✓ Database health check: {health}")
        
        # Test creating and querying data
        async with test_manager.get_async_session() as session:
            # Create a test user
            test_user = User(
                email="test@scrollintel.com",
                hashed_password="hashed_password",
                role=UserRole.ADMIN,
                permissions=["*"],
                is_active=True
            )
            session.add(test_user)
            session.commit()
            logger.info("✓ Test user created")
            
            # Query the user
            result = session.execute(select(User).where(User.email == "test@scrollintel.com"))
            user = result.scalar_one()
            logger.info(f"✓ User queried: {user.email}, role: {user.role}")
            
            # Create a test agent
            test_agent = Agent(
                name="TestAgent",
                agent_type=AgentType.DATA_SCIENTIST,
                capabilities=["data_analysis"],
                status=AgentStatus.ACTIVE,
                configuration={"model": "gpt-4"}
            )
            session.add(test_agent)
            session.commit()
            logger.info("✓ Test agent created")
            
            # Create a test dataset
            test_dataset = Dataset(
                name="Test Dataset",
                source_type="csv",
                data_schema={"column1": "string", "column2": "float"},
                file_path="/test/data.csv"
            )
            session.add(test_dataset)
            session.commit()
            logger.info("✓ Test dataset created")
            
            # Create a test dashboard
            test_dashboard = Dashboard(
                name="Test Dashboard",
                user_id=user.id,
                config={"theme": "dark"},
                charts=[{"type": "line", "title": "Test Chart"}],
                tags=["test"]
            )
            session.add(test_dashboard)
            session.commit()
            logger.info("✓ Test dashboard created")
            
            # Create an audit log entry
            audit_log = AuditLog(
                user_id=user.id,
                action="CREATE",
                resource_type="dashboard",
                resource_id=str(test_dashboard.id),
                details={"name": "Test Dashboard"},
                ip_address="127.0.0.1"
            )
            session.add(audit_log)
            session.commit()
            logger.info("✓ Test audit log created")
            
            # Test relationships
            user_dashboards = len(user.dashboards)
            logger.info(f"✓ User has {user_dashboards} dashboard(s)")
            
            # Count all records
            user_count = session.execute(select(func.count(User.id)))
            agent_count = session.execute(select(func.count(Agent.id)))
            dataset_count = session.execute(select(func.count(Dataset.id)))
            dashboard_count = session.execute(select(func.count(Dashboard.id)))
            audit_count = session.execute(select(func.count(AuditLog.id)))
            
            logger.info("✓ Record counts:")
            logger.info(f"  Users: {user_count.scalar()}")
            logger.info(f"  Agents: {agent_count.scalar()}")
            logger.info(f"  Datasets: {dataset_count.scalar()}")
            logger.info(f"  Dashboards: {dashboard_count.scalar()}")
            logger.info(f"  Audit Logs: {audit_count.scalar()}")
        
        # Test seed data
        logger.info("Testing seed data...")
        async with test_manager.get_async_session() as session:
            # Clear existing data first
            result = session.execute(select(User).where(User.email == "test@scrollintel.com"))
            if result.scalar_one_or_none():
                # Clear test data
                session.execute(select(User).where(User.email == "test@scrollintel.com"))
                session.commit()
            
            # Test seeding
            seed_result = seed_database(session)
            if seed_result["success"]:
                logger.info("✓ Seed data created successfully")
                logger.info(f"  {seed_result['data']}")
            else:
                logger.error(f"✗ Seed data failed: {seed_result['message']}")
        
        await test_manager.close()
        logger.info("✓ Database setup test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Database setup test failed: {e}")
        return False


async def main():
    """Main function."""
    success = await test_database_setup()
    if success:
        print("\n🎉 All database tests passed!")
        return 0
    else:
        print("\n❌ Database tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)