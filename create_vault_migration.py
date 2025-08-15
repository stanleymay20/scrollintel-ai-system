"""
Create Alembic migration for ScrollIntel Vault tables.
Run this script to generate the migration file for vault_insights and vault_access_logs tables.
"""

import subprocess
import sys
from pathlib import Path

def create_migration():
    """Create Alembic migration for vault tables."""
    try:
        # Create migration
        result = subprocess.run([
            sys.executable, "-m", "alembic", "revision", "--autogenerate",
            "-m", "Add vault tables for secure insight storage"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Migration created successfully!")
            print(f"Output: {result.stdout}")
            
            # Show the migration file location
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Generating' in line and '.py' in line:
                    print(f"📁 Migration file: {line.split('Generating ')[1]}")
                    break
        else:
            print("❌ Failed to create migration")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating migration: {e}")
        return False
    
    return True

def run_migration():
    """Run the migration to create the tables."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "alembic", "upgrade", "head"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Migration applied successfully!")
            print(f"Output: {result.stdout}")
        else:
            print("❌ Failed to apply migration")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error applying migration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Creating ScrollIntel Vault migration...")
    
    if create_migration():
        print("\n🔄 Applying migration...")
        if run_migration():
            print("\n✅ Vault tables created successfully!")
            print("\nNew tables:")
            print("  - vault_insights: Secure storage for AI-generated insights")
            print("  - vault_access_logs: Audit trail for insight access")
        else:
            print("\n❌ Failed to apply migration")
    else:
        print("\n❌ Failed to create migration")