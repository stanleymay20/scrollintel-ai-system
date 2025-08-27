#!/bin/bash
# ScrollIntel Automated Backup Script

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="scrollintel_prod"
DB_USER="scrollintel"
DB_HOST="db"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
echo "Creating database backup..."
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_DIR/db_backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/db_backup_$DATE.sql

# Keep only last 7 days of backups
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

# Upload to cloud storage (uncomment and configure)
# aws s3 cp $BACKUP_DIR/db_backup_$DATE.sql.gz s3://your-backup-bucket/

echo "Backup completed: db_backup_$DATE.sql.gz"
