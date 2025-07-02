#!/usr/bin/env python3
"""
Script to run database migrations for enhanced commit fields
"""

import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import asyncio
from alembic.config import Config
from alembic import command
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the database migration"""
    try:
        # Get the alembic configuration
        alembic_cfg_path = backend_dir / "alembic.ini"
        
        if not alembic_cfg_path.exists():
            logger.error(f"alembic.ini not found at {alembic_cfg_path}")
            return False
        
        # Create Alembic configuration
        alembic_cfg = Config(str(alembic_cfg_path))
        
        # Set the script location to the migrations directory
        migrations_dir = backend_dir / "migrations"
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        
        logger.info("Checking current database revision...")
        
        # Check current revision
        try:
            command.current(alembic_cfg, verbose=True)
        except Exception as e:
            logger.warning(f"Could not get current revision: {e}")
        
        logger.info("Running database migration to add enhanced commit fields...")
        
        # Run the migration
        command.upgrade(alembic_cfg, "head")
        
        logger.info("✅ Migration completed successfully!")
        
        # Show new revision
        logger.info("New database revision:")
        command.current(alembic_cfg, verbose=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_columns_exist():
    """Check if the new columns exist in the database"""
    try:
        # Import database components
        from db.database import database, engine
        from sqlalchemy import text
        
        async def check_async():
            try:
                # Connect to database
                await database.connect()
                
                # Check if columns exist
                query = text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'commits' 
                    AND column_name IN ('modified_files', 'file_types', 'modified_directories', 'total_changes', 'change_type', 'commit_size')
                    ORDER BY column_name
                """)
                
                result = await database.fetch_all(query)
                existing_columns = [row[0] for row in result]
                
                expected_columns = ['modified_files', 'file_types', 'modified_directories', 'total_changes', 'change_type', 'commit_size']
                
                logger.info(f"Expected columns: {expected_columns}")
                logger.info(f"Existing columns: {existing_columns}")
                
                missing_columns = set(expected_columns) - set(existing_columns)
                if missing_columns:
                    logger.warning(f"Missing columns: {missing_columns}")
                    return False
                else:
                    logger.info("✅ All enhanced columns are present!")
                    return True
                    
            except Exception as e:
                logger.error(f"Error checking columns: {e}")
                # Try SQLite syntax if PostgreSQL fails
                try:
                    query = text("PRAGMA table_info(commits)")
                    result = await database.fetch_all(query)
                    columns = [row[1] for row in result]  # Column name is at index 1
                    
                    enhanced_columns = ['modified_files', 'file_types', 'modified_directories', 'total_changes', 'change_type', 'commit_size']
                    existing_enhanced = [col for col in enhanced_columns if col in columns]
                    
                    logger.info(f"Found enhanced columns: {existing_enhanced}")
                    return len(existing_enhanced) == len(enhanced_columns)
                    
                except Exception as e2:
                    logger.error(f"Error with SQLite check: {e2}")
                    return False
            finally:
                await database.disconnect()
        
        return asyncio.run(check_async())
        
    except Exception as e:
        logger.error(f"Error in column check: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Enhanced Commit Fields Migration")
    logger.info("=" * 60)
    
    # Check if columns already exist
    logger.info("Checking if enhanced columns already exist...")
    if check_columns_exist():
        logger.info("✅ Enhanced columns already exist! No migration needed.")
        return True
    
    # Run migration
    logger.info("Enhanced columns not found. Running migration...")
    success = run_migration()
    
    if success:
        logger.info("Verifying migration results...")
        if check_columns_exist():
            logger.info("🎉 Migration completed successfully! Enhanced commit tracking is now available.")
        else:
            logger.warning("⚠️  Migration ran but columns verification failed. Please check manually.")
    else:
        logger.error("❌ Migration failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
