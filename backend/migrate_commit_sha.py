"""
Migration script to expand commit SHA columns length
Run this script to update the database schema
"""

import asyncio
import logging
from sqlalchemy import text
from db.database import sync_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Run the migration to expand commit SHA columns"""
    
    try:
        # Use synchronous connection since sync_engine is synchronous
        with sync_engine.connect() as conn:
            logger.info("🔄 Starting migration: expand_commit_sha_length")
            
            # Check current column length
            result = conn.execute(text("""
                SELECT column_name, character_maximum_length 
                FROM information_schema.columns 
                WHERE table_name = 'commits' 
                AND column_name IN ('sha', 'parent_sha')
                ORDER BY column_name;
            """))
            
            current_lengths = {row[0]: row[1] for row in result}
            logger.info(f"📊 Current column lengths: {current_lengths}")
            
            # Only run migration if needed
            if current_lengths.get('sha', 0) < 64:
                logger.info("🔄 Expanding sha column from 40 to 64 characters...")
                conn.execute(text("ALTER TABLE commits ALTER COLUMN sha TYPE VARCHAR(64);"))
                logger.info("✅ sha column expanded successfully")
            else:
                logger.info("✅ sha column already has sufficient length")
                
            if current_lengths.get('parent_sha', 0) < 64:
                logger.info("🔄 Expanding parent_sha column from 40 to 64 characters...")
                conn.execute(text("ALTER TABLE commits ALTER COLUMN parent_sha TYPE VARCHAR(64);"))
                logger.info("✅ parent_sha column expanded successfully")
            else:
                logger.info("✅ parent_sha column already has sufficient length")
            
            # Add comments
            conn.execute(text("""
                COMMENT ON COLUMN commits.sha IS 'Commit SHA hash - supports both SHA-1 (40 chars) and SHA-256 (64 chars)';
            """))
            
            conn.execute(text("""
                COMMENT ON COLUMN commits.parent_sha IS 'Parent commit SHA hash - supports both SHA-1 (40 chars) and SHA-256 (64 chars)';
            """))
            
            logger.info("✅ Comments added successfully")
            
            # Verify the changes
            result = conn.execute(text("""
                SELECT column_name, character_maximum_length 
                FROM information_schema.columns 
                WHERE table_name = 'commits' 
                AND column_name IN ('sha', 'parent_sha')
                ORDER BY column_name;
            """))
            
            new_lengths = {row[0]: row[1] for row in result}
            logger.info(f"📊 New column lengths: {new_lengths}")
            
            # Commit the transaction
            conn.commit()
            logger.info("🎉 Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

def run_migration_sync():
    """Synchronous wrapper for the migration"""
    asyncio.run(run_migration())

if __name__ == "__main__":
    run_migration_sync()
