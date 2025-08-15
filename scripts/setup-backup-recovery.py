#!/usr/bin/env python3

"""
Backup and Disaster Recovery Setup
Configures automated backup and recovery procedures for ScrollIntel
"""

import os
import sys
import json
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError

class BackupManager:
    """Manages backup and disaster recovery operations"""
    
    def __init__(self):
        self.backup_dir = Path("/var/backups/scrollintel")
        self.config_file = Path("/etc/scrollintel/backup-config.json")
        self.log_file = Path("/var/log/scrollintel-backup.log")
        
        # Create directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str) -> None:
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def create_backup_config(self) -> None:
        """Create backup configuration file"""
        self.log("Creating backup configuration...")
        
        config = {
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "scrollintel"),
                "user": os.getenv("DB_USER", "scrollintel"),
                "password": os.getenv("DB_PASSWORD", ""),
                "backup_retention_days": 30
            },
            "files": {
                "directories": [
                    "/var/www/scrollintel",
                    "/etc/nginx",
                    "/etc/ssl",
                    "/var/log/scrollintel",
                    "/etc/scrollintel"
                ],
                "exclude_patterns": [
                    "*.log",
                    "*.tmp",
                    "__pycache__",
                    "node_modules",
                    ".git"
                ],
                "backup_retention_days": 7
            },
            "s3": {
                "bucket": os.getenv("BACKUP_S3_BUCKET", "scrollintel-backups"),
                "region": os.getenv("AWS_REGION", "us-east-1"),
                "access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
                "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
                "encryption": True
            },
            "notifications": {
                "email": os.getenv("BACKUP_EMAIL", "admin@scrollintel.com"),
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
                "on_success": True,
                "on_failure": True
            },
            "schedule": {
                "database_backup": "0 2 * * *",  # Daily at 2 AM
                "file_backup": "0 3 * * 0",     # Weekly on Sunday at 3 AM
                "cleanup": "0 4 * * 0"          # Weekly cleanup on Sunday at 4 AM
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set secure permissions
        os.chmod(self.config_file, 0o600)
        
        self.log(f"Backup configuration created: {self.config_file}")
    
    def load_config(self) -> Dict:
        """Load backup configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.log("Configuration file not found, creating default...")
            self.create_backup_config()
            return self.load_config()
    
    def backup_database(self) -> str:
        """Backup PostgreSQL database"""
        config = self.load_config()
        db_config = config["database"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"scrollintel_db_{timestamp}.sql.gz"
        backup_path = self.backup_dir / backup_filename
        
        self.log(f"Starting database backup: {backup_filename}")
        
        # Set environment variables for pg_dump
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config["password"]
        
        try:
            # Create database dump
            dump_cmd = [
                "pg_dump",
                "-h", db_config["host"],
                "-p", str(db_config["port"]),
                "-U", db_config["user"],
                "-d", db_config["name"],
                "--verbose",
                "--no-password",
                "--format=custom",
                "--compress=9"
            ]
            
            with open(backup_path, 'wb') as f:
                result = subprocess.run(
                    dump_cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=True
                )
            
            # Verify backup
            if backup_path.exists() and backup_path.stat().st_size > 0:
                self.log(f"Database backup completed: {backup_path}")
                return str(backup_path)
            else:
                raise Exception("Backup file is empty or doesn't exist")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Database backup failed: {e.stderr.decode()}"
            self.log(error_msg)
            raise Exception(error_msg)
    
    def backup_files(self) -> str:
        """Backup application files"""
        config = self.load_config()
        file_config = config["files"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"scrollintel_files_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_filename
        
        self.log(f"Starting file backup: {backup_filename}")
        
        try:
            # Create tar command with exclusions
            tar_cmd = ["tar", "-czf", str(backup_path)]
            
            # Add exclusion patterns
            for pattern in file_config["exclude_patterns"]:
                tar_cmd.extend(["--exclude", pattern])
            
            # Add directories to backup
            tar_cmd.extend(file_config["directories"])
            
            # Execute backup
            result = subprocess.run(
                tar_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Verify backup
            if backup_path.exists() and backup_path.stat().st_size > 0:
                self.log(f"File backup completed: {backup_path}")
                return str(backup_path)
            else:
                raise Exception("Backup file is empty or doesn't exist")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"File backup failed: {e.stderr}"
            self.log(error_msg)
            raise Exception(error_msg)
    
    def upload_to_s3(self, backup_path: str) -> bool:
        """Upload backup to S3"""
        config = self.load_config()
        s3_config = config["s3"]
        
        if not all([s3_config["access_key"], s3_config["secret_key"], s3_config["bucket"]]):
            self.log("S3 configuration incomplete, skipping upload")
            return False
        
        self.log(f"Uploading to S3: {backup_path}")
        
        try:
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config["access_key"],
                aws_secret_access_key=s3_config["secret_key"],
                region_name=s3_config["region"]
            )
            
            # Upload file
            backup_file = Path(backup_path)
            s3_key = f"backups/{backup_file.name}"
            
            extra_args = {}
            if s3_config["encryption"]:
                extra_args["ServerSideEncryption"] = "AES256"
            
            s3_client.upload_file(
                str(backup_path),
                s3_config["bucket"],
                s3_key,
                ExtraArgs=extra_args
            )
            
            self.log(f"Successfully uploaded to S3: s3://{s3_config['bucket']}/{s3_key}")
            return True
            
        except ClientError as e:
            error_msg = f"S3 upload failed: {e}"
            self.log(error_msg)
            return False
    
    def cleanup_old_backups(self) -> None:
        """Clean up old backup files"""
        config = self.load_config()
        
        self.log("Starting backup cleanup...")
        
        # Clean up local database backups
        db_retention_days = config["database"]["backup_retention_days"]
        db_cutoff_date = datetime.now() - timedelta(days=db_retention_days)
        
        for backup_file in self.backup_dir.glob("scrollintel_db_*.sql.gz"):
            if backup_file.stat().st_mtime < db_cutoff_date.timestamp():
                backup_file.unlink()
                self.log(f"Deleted old database backup: {backup_file.name}")
        
        # Clean up local file backups
        file_retention_days = config["files"]["backup_retention_days"]
        file_cutoff_date = datetime.now() - timedelta(days=file_retention_days)
        
        for backup_file in self.backup_dir.glob("scrollintel_files_*.tar.gz"):
            if backup_file.stat().st_mtime < file_cutoff_date.timestamp():
                backup_file.unlink()
                self.log(f"Deleted old file backup: {backup_file.name}")
        
        self.log("Backup cleanup completed")
    
    def send_notification(self, subject: str, message: str, is_success: bool = True) -> None:
        """Send backup notification"""
        config = self.load_config()
        notifications = config["notifications"]
        
        if (is_success and not notifications["on_success"]) or \
           (not is_success and not notifications["on_failure"]):
            return
        
        # Send email notification
        if notifications["email"]:
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                # Configure SMTP (adjust for your email provider)
                smtp_server = os.getenv("SMTP_SERVER", "localhost")
                smtp_port = int(os.getenv("SMTP_PORT", "587"))
                smtp_user = os.getenv("SMTP_USER", "")
                smtp_password = os.getenv("SMTP_PASSWORD", "")
                
                msg = MIMEMultipart()
                msg['From'] = smtp_user
                msg['To'] = notifications["email"]
                msg['Subject'] = f"ScrollIntel Backup: {subject}"
                
                msg.attach(MIMEText(message, 'plain'))
                
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                server.quit()
                
                self.log("Email notification sent")
                
            except Exception as e:
                self.log(f"Failed to send email notification: {e}")
        
        # Send Slack notification
        if notifications["slack_webhook"]:
            try:
                import requests
                
                emoji = "✅" if is_success else "❌"
                payload = {
                    "text": f"{emoji} ScrollIntel Backup: {subject}",
                    "attachments": [
                        {
                            "color": "good" if is_success else "danger",
                            "text": message
                        }
                    ]
                }
                
                response = requests.post(
                    notifications["slack_webhook"],
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()
                
                self.log("Slack notification sent")
                
            except Exception as e:
                self.log(f"Failed to send Slack notification: {e}")
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        config = self.load_config()
        db_config = config["database"]
        
        self.log(f"Starting database restore from: {backup_path}")
        
        # Set environment variables for pg_restore
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config["password"]
        
        try:
            # Drop existing database (be careful!)
            drop_cmd = [
                "dropdb",
                "-h", db_config["host"],
                "-p", str(db_config["port"]),
                "-U", db_config["user"],
                "--if-exists",
                db_config["name"]
            ]
            
            subprocess.run(drop_cmd, env=env, check=True)
            
            # Create new database
            create_cmd = [
                "createdb",
                "-h", db_config["host"],
                "-p", str(db_config["port"]),
                "-U", db_config["user"],
                db_config["name"]
            ]
            
            subprocess.run(create_cmd, env=env, check=True)
            
            # Restore from backup
            restore_cmd = [
                "pg_restore",
                "-h", db_config["host"],
                "-p", str(db_config["port"]),
                "-U", db_config["user"],
                "-d", db_config["name"],
                "--verbose",
                "--no-password",
                backup_path
            ]
            
            result = subprocess.run(
                restore_cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.log("Database restore completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Database restore failed: {e.stderr}"
            self.log(error_msg)
            return False
    
    def create_cron_jobs(self) -> None:
        """Create cron jobs for automated backups"""
        config = self.load_config()
        schedule = config["schedule"]
        
        self.log("Creating cron jobs...")
        
        # Create backup scripts
        db_backup_script = Path("/usr/local/bin/scrollintel-db-backup.sh")
        file_backup_script = Path("/usr/local/bin/scrollintel-file-backup.sh")
        cleanup_script = Path("/usr/local/bin/scrollintel-backup-cleanup.sh")
        
        # Database backup script
        with open(db_backup_script, 'w') as f:
            f.write(f"""#!/bin/bash
cd {Path(__file__).parent}
python3 {__file__} --database-backup
""")
        
        # File backup script
        with open(file_backup_script, 'w') as f:
            f.write(f"""#!/bin/bash
cd {Path(__file__).parent}
python3 {__file__} --file-backup
""")
        
        # Cleanup script
        with open(cleanup_script, 'w') as f:
            f.write(f"""#!/bin/bash
cd {Path(__file__).parent}
python3 {__file__} --cleanup
""")
        
        # Make scripts executable
        for script in [db_backup_script, file_backup_script, cleanup_script]:
            os.chmod(script, 0o755)
        
        # Add cron jobs
        cron_entries = [
            f"{schedule['database_backup']} root {db_backup_script}",
            f"{schedule['file_backup']} root {file_backup_script}",
            f"{schedule['cleanup']} root {cleanup_script}"
        ]
        
        cron_file = Path("/etc/cron.d/scrollintel-backup")
        with open(cron_file, 'w') as f:
            f.write("# ScrollIntel Backup Cron Jobs\n")
            f.write("SHELL=/bin/bash\n")
            f.write("PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin\n\n")
            for entry in cron_entries:
                f.write(f"{entry}\n")
        
        self.log("Cron jobs created successfully")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ScrollIntel Backup and Recovery")
    parser.add_argument("--database-backup", action="store_true", help="Perform database backup")
    parser.add_argument("--file-backup", action="store_true", help="Perform file backup")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old backups")
    parser.add_argument("--restore-db", help="Restore database from backup file")
    parser.add_argument("--setup", action="store_true", help="Setup backup system")
    
    args = parser.parse_args()
    
    manager = BackupManager()
    
    try:
        if args.setup:
            print("🔧 Setting up backup and recovery system...")
            manager.create_backup_config()
            manager.create_cron_jobs()
            print("✅ Backup system setup completed!")
            
        elif args.database_backup:
            backup_path = manager.backup_database()
            if manager.upload_to_s3(backup_path):
                manager.send_notification(
                    "Database Backup Successful",
                    f"Database backup completed and uploaded to S3: {backup_path}"
                )
            else:
                manager.send_notification(
                    "Database Backup Completed (Local Only)",
                    f"Database backup completed locally: {backup_path}",
                    is_success=False
                )
                
        elif args.file_backup:
            backup_path = manager.backup_files()
            if manager.upload_to_s3(backup_path):
                manager.send_notification(
                    "File Backup Successful",
                    f"File backup completed and uploaded to S3: {backup_path}"
                )
            else:
                manager.send_notification(
                    "File Backup Completed (Local Only)",
                    f"File backup completed locally: {backup_path}",
                    is_success=False
                )
                
        elif args.cleanup:
            manager.cleanup_old_backups()
            
        elif args.restore_db:
            if manager.restore_database(args.restore_db):
                manager.send_notification(
                    "Database Restore Successful",
                    f"Database restored from: {args.restore_db}"
                )
            else:
                manager.send_notification(
                    "Database Restore Failed",
                    f"Failed to restore database from: {args.restore_db}",
                    is_success=False
                )
        else:
            parser.print_help()
            
    except Exception as e:
        manager.log(f"Operation failed: {e}")
        manager.send_notification(
            "Backup Operation Failed",
            f"Error: {e}",
            is_success=False
        )
        sys.exit(1)

if __name__ == "__main__":
    main()