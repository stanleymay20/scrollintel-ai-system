#!/usr/bin/env python3
"""
ScrollIntel Database Replication Setup
Sets up PostgreSQL master-slave replication for high availability
"""

import os
import sys
import time
import logging
import subprocess
import psycopg2
from typing import Dict, List, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseReplicationManager:
    def __init__(self):
        self.config = {
            'master_host': os.getenv('DB_MASTER_HOST', 'localhost'),
            'master_port': int(os.getenv('DB_MASTER_PORT', '5432')),
            'replica_host': os.getenv('DB_REPLICA_HOST', 'localhost'),
            'replica_port': int(os.getenv('DB_REPLICA_PORT', '5433')),
            'database': os.getenv('POSTGRES_DB', 'scrollintel'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'replication_user': os.getenv('REPLICATION_USER', 'replicator'),
            'replication_password': os.getenv('REPLICATION_PASSWORD'),
        }
        
        if not self.config['password'] or not self.config['replication_password']:
            logger.error("Database passwords must be set via environment variables")
            sys.exit(1)
        
        logger.info("Database replication manager initialized")

    def create_replication_user(self):
        """Create replication user on master database"""
        logger.info("Creating replication user on master database...")
        
        try:
            conn = psycopg2.connect(
                host=self.config['master_host'],
                port=self.config['master_port'],
                database=self.config['database'],
                user=self.config['username'],
                password=self.config['password']
            )
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                # Check if replication user exists
                cursor.execute(
                    "SELECT 1 FROM pg_roles WHERE rolname = %s",
                    (self.config['replication_user'],)
                )
                
                if not cursor.fetchone():
                    # Create replication user
                    cursor.execute(f"""
                        CREATE USER {self.config['replication_user']} 
                        WITH REPLICATION ENCRYPTED PASSWORD %s
                    """, (self.config['replication_password'],))
                    
                    logger.info(f"Created replication user: {self.config['replication_user']}")
                else:
                    logger.info("Replication user already exists")
            
            conn.close()
            
        except psycopg2.Error as e:
            logger.error(f"Failed to create replication user: {e}")
            raise

    def configure_master_database(self):
        """Configure master database for replication"""
        logger.info("Configuring master database for replication...")
        
        # PostgreSQL configuration for master
        postgresql_conf = f"""
# Replication settings
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
synchronous_commit = on
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/archive/%f'

# Connection settings
listen_addresses = '*'
port = {self.config['master_port']}

# Performance settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'
log_min_duration_statement = 1000
"""
        
        # pg_hba.conf configuration
        pg_hba_conf = f"""
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             all                                     trust
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5

# Replication connections
host    replication     {self.config['replication_user']}     {self.config['replica_host']}/32    md5
host    replication     {self.config['replication_user']}     0.0.0.0/0                           md5

# Application connections
host    {self.config['database']}    {self.config['username']}    0.0.0.0/0    md5
"""
        
        try:
            # Write configuration files
            os.makedirs('./postgres-config/master', exist_ok=True)
            
            with open('./postgres-config/master/postgresql.conf', 'w') as f:
                f.write(postgresql_conf)
            
            with open('./postgres-config/master/pg_hba.conf', 'w') as f:
                f.write(pg_hba_conf)
            
            logger.info("Master database configuration files created")
            
        except IOError as e:
            logger.error(f"Failed to write configuration files: {e}")
            raise

    def configure_replica_database(self):
        """Configure replica database"""
        logger.info("Configuring replica database...")
        
        # PostgreSQL configuration for replica
        postgresql_conf = f"""
# Replication settings
hot_standby = on
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s
hot_standby_feedback = on

# Connection settings
listen_addresses = '*'
port = {self.config['replica_port']}

# Performance settings (read-only optimized)
shared_buffers = 256MB
effective_cache_size = 1GB
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-replica-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000
"""
        
        # recovery.conf for replica
        recovery_conf = f"""
standby_mode = 'on'
primary_conninfo = 'host={self.config['master_host']} port={self.config['master_port']} user={self.config['replication_user']} password={self.config['replication_password']}'
trigger_file = '/tmp/postgresql.trigger'
"""
        
        try:
            # Write configuration files
            os.makedirs('./postgres-config/replica', exist_ok=True)
            
            with open('./postgres-config/replica/postgresql.conf', 'w') as f:
                f.write(postgresql_conf)
            
            with open('./postgres-config/replica/recovery.conf', 'w') as f:
                f.write(recovery_conf)
            
            logger.info("Replica database configuration files created")
            
        except IOError as e:
            logger.error(f"Failed to write replica configuration files: {e}")
            raise

    def create_docker_compose_replication(self):
        """Create Docker Compose configuration for database replication"""
        logger.info("Creating Docker Compose configuration for replication...")
        
        compose_config = {
            'version': '3.8',
            'services': {
                'postgres-master': {
                    'image': 'postgres:15-alpine',
                    'container_name': 'scrollintel-postgres-master',
                    'environment': {
                        'POSTGRES_DB': self.config['database'],
                        'POSTGRES_USER': self.config['username'],
                        'POSTGRES_PASSWORD': self.config['password'],
                        'POSTGRES_INITDB_ARGS': '--auth-host=md5'
                    },
                    'volumes': [
                        'postgres_master_data:/var/lib/postgresql/data',
                        './postgres-config/master/postgresql.conf:/etc/postgresql/postgresql.conf',
                        './postgres-config/master/pg_hba.conf:/etc/postgresql/pg_hba.conf',
                        'postgres_archive:/var/lib/postgresql/archive'
                    ],
                    'ports': [f"{self.config['master_port']}:5432"],
                    'networks': ['scrollintel-db-network'],
                    'command': [
                        'postgres',
                        '-c', 'config_file=/etc/postgresql/postgresql.conf',
                        '-c', 'hba_file=/etc/postgresql/pg_hba.conf'
                    ],
                    'healthcheck': {
                        'test': ['CMD-SHELL', f"pg_isready -U {self.config['username']}"],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 5
                    },
                    'restart': 'unless-stopped'
                },
                'postgres-replica': {
                    'image': 'postgres:15-alpine',
                    'container_name': 'scrollintel-postgres-replica',
                    'environment': {
                        'POSTGRES_DB': self.config['database'],
                        'POSTGRES_USER': self.config['username'],
                        'POSTGRES_PASSWORD': self.config['password'],
                        'PGUSER': self.config['username']
                    },
                    'volumes': [
                        'postgres_replica_data:/var/lib/postgresql/data',
                        './postgres-config/replica/postgresql.conf:/etc/postgresql/postgresql.conf',
                        './postgres-config/replica/recovery.conf:/var/lib/postgresql/data/recovery.conf'
                    ],
                    'ports': [f"{self.config['replica_port']}:5432"],
                    'networks': ['scrollintel-db-network'],
                    'depends_on': {
                        'postgres-master': {
                            'condition': 'service_healthy'
                        }
                    },
                    'command': [
                        'postgres',
                        '-c', 'config_file=/etc/postgresql/postgresql.conf'
                    ],
                    'healthcheck': {
                        'test': ['CMD-SHELL', f"pg_isready -U {self.config['username']}"],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 5
                    },
                    'restart': 'unless-stopped'
                },
                'pgbouncer': {
                    'image': 'pgbouncer/pgbouncer:latest',
                    'container_name': 'scrollintel-pgbouncer',
                    'environment': {
                        'DATABASES_HOST': 'postgres-master',
                        'DATABASES_PORT': '5432',
                        'DATABASES_USER': self.config['username'],
                        'DATABASES_PASSWORD': self.config['password'],
                        'DATABASES_DBNAME': self.config['database'],
                        'POOL_MODE': 'transaction',
                        'MAX_CLIENT_CONN': '100',
                        'DEFAULT_POOL_SIZE': '25'
                    },
                    'ports': ['6432:5432'],
                    'networks': ['scrollintel-db-network'],
                    'depends_on': ['postgres-master'],
                    'restart': 'unless-stopped'
                }
            },
            'volumes': {
                'postgres_master_data': None,
                'postgres_replica_data': None,
                'postgres_archive': None
            },
            'networks': {
                'scrollintel-db-network': {
                    'driver': 'bridge'
                }
            }
        }
        
        try:
            with open('docker-compose.db-replication.yml', 'w') as f:
                yaml.dump(compose_config, f, default_flow_style=False)
            
            logger.info("Docker Compose replication configuration created")
            
        except IOError as e:
            logger.error(f"Failed to create Docker Compose configuration: {e}")
            raise

    def setup_base_backup(self):
        """Create base backup for replica initialization"""
        logger.info("Creating base backup for replica initialization...")
        
        try:
            # Ensure master is running
            self.wait_for_master()
            
            # Create base backup
            backup_dir = './postgres-backup'
            os.makedirs(backup_dir, exist_ok=True)
            
            subprocess.run([
                'docker', 'exec', 'scrollintel-postgres-master',
                'pg_basebackup',
                '-h', 'localhost',
                '-D', '/tmp/backup',
                '-U', self.config['replication_user'],
                '-v', '-P', '-W'
            ], check=True, input=self.config['replication_password'], text=True)
            
            # Copy backup from container
            subprocess.run([
                'docker', 'cp',
                'scrollintel-postgres-master:/tmp/backup',
                backup_dir
            ], check=True)
            
            logger.info("Base backup created successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Base backup failed: {e}")
            raise

    def wait_for_master(self, timeout: int = 60):
        """Wait for master database to be ready"""
        logger.info("Waiting for master database to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                conn = psycopg2.connect(
                    host=self.config['master_host'],
                    port=self.config['master_port'],
                    database=self.config['database'],
                    user=self.config['username'],
                    password=self.config['password'],
                    connect_timeout=5
                )
                conn.close()
                logger.info("Master database is ready")
                return
                
            except psycopg2.Error:
                time.sleep(5)
        
        raise TimeoutError("Master database did not become ready within timeout")

    def verify_replication(self):
        """Verify that replication is working correctly"""
        logger.info("Verifying replication setup...")
        
        try:
            # Connect to master
            master_conn = psycopg2.connect(
                host=self.config['master_host'],
                port=self.config['master_port'],
                database=self.config['database'],
                user=self.config['username'],
                password=self.config['password']
            )
            
            # Connect to replica
            replica_conn = psycopg2.connect(
                host=self.config['replica_host'],
                port=self.config['replica_port'],
                database=self.config['database'],
                user=self.config['username'],
                password=self.config['password']
            )
            
            # Create test table on master
            with master_conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS replication_test")
                cursor.execute("""
                    CREATE TABLE replication_test (
                        id SERIAL PRIMARY KEY,
                        test_data TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                cursor.execute(
                    "INSERT INTO replication_test (test_data) VALUES (%s)",
                    ("Replication test data",)
                )
                master_conn.commit()
            
            # Wait for replication
            time.sleep(5)
            
            # Check if data exists on replica
            with replica_conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM replication_test")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.info("✅ Replication verification successful")
                    
                    # Check replication status
                    cursor.execute("""
                        SELECT 
                            client_addr,
                            state,
                            sent_lsn,
                            write_lsn,
                            flush_lsn,
                            replay_lsn
                        FROM pg_stat_replication
                    """)
                    
                    replication_stats = cursor.fetchall()
                    if replication_stats:
                        logger.info("Replication status:")
                        for stat in replication_stats:
                            logger.info(f"  Client: {stat[0]}, State: {stat[1]}")
                    
                    return True
                else:
                    logger.error("❌ Replication verification failed - no data found on replica")
                    return False
            
        except psycopg2.Error as e:
            logger.error(f"Replication verification failed: {e}")
            return False
        
        finally:
            if 'master_conn' in locals():
                master_conn.close()
            if 'replica_conn' in locals():
                replica_conn.close()

    def create_monitoring_queries(self):
        """Create SQL queries for monitoring replication"""
        logger.info("Creating replication monitoring queries...")
        
        monitoring_queries = {
            'replication_status.sql': """
-- Check replication status on master
SELECT 
    client_addr,
    client_hostname,
    client_port,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    write_lag,
    flush_lag,
    replay_lag,
    sync_state
FROM pg_stat_replication;
""",
            'replication_lag.sql': """
-- Check replication lag
SELECT 
    CASE 
        WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn() 
        THEN 0 
        ELSE EXTRACT(EPOCH FROM now() - pg_last_xact_replay_timestamp()) 
    END AS lag_seconds;
""",
            'replica_status.sql': """
-- Check replica status
SELECT 
    pg_is_in_recovery() as is_replica,
    pg_last_wal_receive_lsn() as receive_lsn,
    pg_last_wal_replay_lsn() as replay_lsn,
    pg_last_xact_replay_timestamp() as last_replay;
"""
        }
        
        os.makedirs('./monitoring/sql', exist_ok=True)
        
        for filename, query in monitoring_queries.items():
            with open(f'./monitoring/sql/{filename}', 'w') as f:
                f.write(query)
        
        logger.info("Monitoring queries created in ./monitoring/sql/")

    def setup_replication(self):
        """Complete replication setup process"""
        logger.info("Starting database replication setup...")
        
        try:
            # Step 1: Configure master and replica
            self.configure_master_database()
            self.configure_replica_database()
            
            # Step 2: Create Docker Compose configuration
            self.create_docker_compose_replication()
            
            # Step 3: Start master database
            logger.info("Starting master database...")
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.db-replication.yml',
                'up', '-d', 'postgres-master'
            ], check=True)
            
            # Step 4: Wait for master and create replication user
            self.wait_for_master()
            self.create_replication_user()
            
            # Step 5: Start replica
            logger.info("Starting replica database...")
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.db-replication.yml',
                'up', '-d', 'postgres-replica'
            ], check=True)
            
            # Step 6: Wait for replica to be ready
            time.sleep(30)
            
            # Step 7: Verify replication
            if self.verify_replication():
                logger.info("✅ Database replication setup completed successfully!")
                
                # Step 8: Create monitoring queries
                self.create_monitoring_queries()
                
                # Step 9: Start PgBouncer
                logger.info("Starting PgBouncer connection pooler...")
                subprocess.run([
                    'docker-compose', '-f', 'docker-compose.db-replication.yml',
                    'up', '-d', 'pgbouncer'
                ], check=True)
                
                logger.info("Database replication setup completed!")
                logger.info(f"Master: {self.config['master_host']}:{self.config['master_port']}")
                logger.info(f"Replica: {self.config['replica_host']}:{self.config['replica_port']}")
                logger.info("PgBouncer: localhost:6432")
                
                return True
            else:
                logger.error("❌ Replication verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Replication setup failed: {e}")
            return False

if __name__ == '__main__':
    manager = DatabaseReplicationManager()
    success = manager.setup_replication()
    sys.exit(0 if success else 1)