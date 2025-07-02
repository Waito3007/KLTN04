#!/usr/bin/env python3
"""
Direct SQL approach to add enhanced commit fields
Use this if Alembic migration fails
"""

import os
import sys
import asyncio
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def add_enhanced_fields_direct():
    """Add enhanced fields directly using SQL"""
    try:
        from db.database import database, engine
        
        # Connect to database
        await database.connect()
        
        # SQL commands to add columns
        sql_commands = [
            "ALTER TABLE commits ADD COLUMN modified_files TEXT",
            "ALTER TABLE commits ADD COLUMN file_types TEXT", 
            "ALTER TABLE commits ADD COLUMN modified_directories TEXT",
            "ALTER TABLE commits ADD COLUMN total_changes INTEGER",
            "ALTER TABLE commits ADD COLUMN change_type VARCHAR(50)",
            "ALTER TABLE commits ADD COLUMN commit_size VARCHAR(20)"
        ]
        
        logger.info("Adding enhanced commit fields...")
        
        for sql_command in sql_commands:
            try:
                logger.info(f"Executing: {sql_command}")
                await database.execute(text(sql_command))
                logger.info("✅ Success")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                    logger.info("⚠️  Column already exists, skipping")
                else:
                    logger.error(f"❌ Failed: {e}")
                    return False
        
        logger.info("🎉 All enhanced fields added successfully!")
        
        # Verify columns exist
        try:
            # Test query to make sure columns are accessible
            test_query = text("""
                SELECT modified_files, file_types, modified_directories, 
                       total_changes, change_type, commit_size 
                FROM commits LIMIT 1
            """)
            await database.fetch_one(test_query)
            logger.info("✅ Column verification successful!")
            
        except Exception as e:
            logger.warning(f"⚠️  Column verification failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error adding enhanced fields: {e}")
        return False
    finally:
        await database.disconnect()

async def check_and_fix_columns():
    """Check existing columns and add missing ones"""
    try:
        from db.database import database
        
        await database.connect()
        
        # Get existing columns
        try:
            # Try PostgreSQL syntax first
            query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'commits'
                ORDER BY column_name
            """)
            result = await database.fetch_all(query)
            existing_columns = [row[0] for row in result]
        except:
            # Try SQLite syntax
            query = text("PRAGMA table_info(commits)")
            result = await database.fetch_all(query)
            existing_columns = [row[1] for row in result]  # Column name is at index 1
        
        logger.info(f"Existing columns: {existing_columns}")
        
        # Check which enhanced columns are missing
        enhanced_columns = {
            'modified_files': 'TEXT',
            'file_types': 'TEXT', 
            'modified_directories': 'TEXT',
            'total_changes': 'INTEGER',
            'change_type': 'VARCHAR(50)',
            'commit_size': 'VARCHAR(20)'
        }
        
        missing_columns = []
        for col_name in enhanced_columns:
            if col_name not in existing_columns:
                missing_columns.append(col_name)
        
        if not missing_columns:
            logger.info("✅ All enhanced columns already exist!")
            return True
        
        logger.info(f"Missing columns: {missing_columns}")
        
        # Add missing columns
        for col_name in missing_columns:
            col_type = enhanced_columns[col_name]
            sql_command = f"ALTER TABLE commits ADD COLUMN {col_name} {col_type}"
            
            try:
                logger.info(f"Adding column: {col_name}")
                await database.execute(text(sql_command))
                logger.info(f"✅ Added {col_name}")
            except Exception as e:
                logger.error(f"❌ Failed to add {col_name}: {e}")
                return False
        
        logger.info("🎉 All missing columns added successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in check_and_fix_columns: {e}")
        return False
    finally:
        await database.disconnect()

def main():
    """Main function"""
    logger.info("🚀 Direct SQL Enhanced Fields Addition")
    logger.info("=" * 50)
    
    try:
        # Check and add missing columns
        success = asyncio.run(check_and_fix_columns())
        
        if success:
            logger.info("✅ Enhanced commit fields are now available!")
            logger.info("You can now restart your FastAPI application.")
        else:
            logger.error("❌ Failed to add enhanced fields.")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(1)
