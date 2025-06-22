# backend/scripts/migrate_collaborators.py
"""
Migration script to refactor collaborators model and migrate existing data
"""

import asyncio
import logging
from sqlalchemy import select, insert, update, text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Index
from db.database import database, engine
from db.models.repository_collaborators import repository_collaborators
from db.models.users import users
from db.models.repositories import repositories
from datetime import datetime

logger = logging.getLogger(__name__)

async def create_new_collaborators_table():
    """Create the new collaborators table with enhanced schema"""
    try:
        # Create the new collaborators table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS collaborators_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            github_user_id INTEGER UNIQUE NOT NULL,
            github_username VARCHAR(255) NOT NULL,
            display_name VARCHAR(255),
            email VARCHAR(255),
            avatar_url VARCHAR(500),
            bio TEXT,
            company VARCHAR(255),
            location VARCHAR(255),
            blog VARCHAR(500),
            is_site_admin BOOLEAN DEFAULT FALSE,
            node_id VARCHAR(255),
            gravatar_id VARCHAR(255),
            type VARCHAR(50) DEFAULT 'User',
            user_id INTEGER,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
        
        await database.execute(text(create_table_sql))
        
        # Create indexes
        await database.execute(text("CREATE INDEX IF NOT EXISTS idx_collaborators_github_user_id ON collaborators_new(github_user_id);"))
        await database.execute(text("CREATE INDEX IF NOT EXISTS idx_collaborators_github_username ON collaborators_new(github_username);"))
        
        logger.info("Created new collaborators table")
        
    except Exception as e:
        logger.error(f"Error creating new collaborators table: {e}")
        raise

async def migrate_existing_data():
    """Migrate existing data from users and repository_collaborators to new schema"""
    try:
        # First, migrate unique GitHub users from repository_collaborators to collaborators_new
        migrate_sql = """
        INSERT OR IGNORE INTO collaborators_new (
            github_user_id, github_username, display_name, email, avatar_url, 
            bio, company, location, blog, user_id, created_at, updated_at
        )
        SELECT DISTINCT 
            u.github_id,
            u.github_username,
            u.display_name,
            u.email,
            u.avatar_url,
            u.bio,
            u.company,
            u.location,
            u.blog,
            u.id,
            COALESCE(u.created_at, CURRENT_TIMESTAMP),
            CURRENT_TIMESTAMP
        FROM repository_collaborators rc
        JOIN users u ON rc.user_id = u.id
        WHERE u.github_id IS NOT NULL;
        """
        
        result = await database.execute(text(migrate_sql))
        logger.info(f"Migrated {result} unique collaborators from existing data")
        
        return result
        
    except Exception as e:
        logger.error(f"Error migrating existing data: {e}")
        raise

async def create_new_repository_collaborators_table():
    """Create new repository_collaborators table with collaborator_id reference"""
    try:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS repository_collaborators_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repository_id INTEGER NOT NULL,
            collaborator_id INTEGER NOT NULL,
            role VARCHAR(50) NOT NULL,
            permissions VARCHAR(100),
            is_owner BOOLEAN NOT NULL DEFAULT FALSE,
            joined_at DATETIME,
            invited_by VARCHAR(255),
            invitation_status VARCHAR(20),
            commits_count INTEGER DEFAULT 0,
            issues_count INTEGER DEFAULT 0,
            prs_count INTEGER DEFAULT 0,
            last_activity DATETIME,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_synced DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (repository_id) REFERENCES repositories(id),
            FOREIGN KEY (collaborator_id) REFERENCES collaborators_new(id)
        );
        """
        
        await database.execute(text(create_table_sql))
        
        # Create indexes
        await database.execute(text("CREATE INDEX IF NOT EXISTS idx_repo_collaborators_repo_new ON repository_collaborators_new(repository_id);"))
        await database.execute(text("CREATE INDEX IF NOT EXISTS idx_repo_collaborators_collaborator_new ON repository_collaborators_new(collaborator_id);"))
        await database.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_repo_collaborators_unique_new ON repository_collaborators_new(repository_id, collaborator_id);"))
        
        logger.info("Created new repository_collaborators table")
        
    except Exception as e:
        logger.error(f"Error creating new repository_collaborators table: {e}")
        raise

async def migrate_repository_collaborators():
    """Migrate repository_collaborators to use collaborator_id instead of user_id"""
    try:
        migrate_sql = """
        INSERT OR IGNORE INTO repository_collaborators_new (
            repository_id, collaborator_id, role, permissions, is_owner,
            joined_at, invited_by, invitation_status, commits_count, 
            issues_count, prs_count, last_activity, created_at, updated_at, last_synced
        )
        SELECT 
            rc.repository_id,
            c.id as collaborator_id,
            rc.role,
            rc.permissions,
            COALESCE(rc.is_owner, FALSE),
            rc.joined_at,
            rc.invited_by,
            rc.invitation_status,
            COALESCE(rc.commits_count, 0),
            COALESCE(rc.issues_count, 0),
            COALESCE(rc.prs_count, 0),
            rc.last_activity,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP,
            COALESCE(rc.last_synced, CURRENT_TIMESTAMP)
        FROM repository_collaborators rc
        JOIN users u ON rc.user_id = u.id
        JOIN collaborators_new c ON u.github_id = c.github_user_id;
        """
        
        result = await database.execute(text(migrate_sql))
        logger.info(f"Migrated {result} repository collaborator relationships")
        
        return result
        
    except Exception as e:
        logger.error(f"Error migrating repository collaborators: {e}")
        raise

async def backup_old_tables():
    """Backup old tables before dropping them"""
    try:
        # Backup old tables
        await database.execute(text("CREATE TABLE IF NOT EXISTS collaborators_backup AS SELECT * FROM collaborators;"))
        await database.execute(text("CREATE TABLE IF NOT EXISTS repository_collaborators_backup AS SELECT * FROM repository_collaborators;"))
        
        logger.info("Created backup tables")
        
    except Exception as e:
        logger.error(f"Error creating backup tables: {e}")
        raise

async def replace_tables():
    """Replace old tables with new ones"""
    try:
        # Drop old tables
        await database.execute(text("DROP TABLE IF EXISTS collaborators;"))
        await database.execute(text("DROP TABLE IF EXISTS repository_collaborators;"))
        
        # Rename new tables
        await database.execute(text("ALTER TABLE collaborators_new RENAME TO collaborators;"))
        await database.execute(text("ALTER TABLE repository_collaborators_new RENAME TO repository_collaborators;"))
        
        logger.info("Replaced old tables with new ones")
        
    except Exception as e:
        logger.error(f"Error replacing tables: {e}")
        raise

async def verify_migration():
    """Verify the migration was successful"""
    try:
        # Count records in new tables
        collaborators_count = await database.fetch_val(text("SELECT COUNT(*) FROM collaborators;"))
        repo_collabs_count = await database.fetch_val(text("SELECT COUNT(*) FROM repository_collaborators;"))
        
        logger.info(f"Migration verification:")
        logger.info(f"  - Collaborators: {collaborators_count} records")
        logger.info(f"  - Repository collaborators: {repo_collabs_count} records")
        
        # Test a join query
        test_query = text("""
            SELECT COUNT(*) FROM repository_collaborators rc
            JOIN collaborators c ON rc.collaborator_id = c.id
            JOIN repositories r ON rc.repository_id = r.id;
        """)
        join_count = await database.fetch_val(test_query)
        logger.info(f"  - Successful joins: {join_count} records")
        
        return {
            'collaborators_count': collaborators_count,
            'repo_collabs_count': repo_collabs_count,
            'join_count': join_count
        }
        
    except Exception as e:
        logger.error(f"Error verifying migration: {e}")
        raise

async def run_migration():
    """Run the complete migration process"""
    try:
        logger.info("Starting collaborators migration...")
        
        # Connect to database
        await database.connect()
        
        # Step 1: Backup old tables
        logger.info("Step 1: Creating backups...")
        await backup_old_tables()
        
        # Step 2: Create new collaborators table
        logger.info("Step 2: Creating new collaborators table...")
        await create_new_collaborators_table()
        
        # Step 3: Migrate existing data to new collaborators table
        logger.info("Step 3: Migrating collaborator data...")
        collaborators_migrated = await migrate_existing_data()
        
        # Step 4: Create new repository_collaborators table
        logger.info("Step 4: Creating new repository_collaborators table...")
        await create_new_repository_collaborators_table()
        
        # Step 5: Migrate repository_collaborators relationships
        logger.info("Step 5: Migrating repository collaborator relationships...")
        relationships_migrated = await migrate_repository_collaborators()
        
        # Step 6: Replace old tables
        logger.info("Step 6: Replacing old tables...")
        await replace_tables()
        
        # Step 7: Verify migration
        logger.info("Step 7: Verifying migration...")
        verification = await verify_migration()
        
        logger.info("Migration completed successfully!")
        logger.info(f"Summary: {collaborators_migrated} collaborators, {relationships_migrated} relationships migrated")
        logger.info(f"Verification: {verification}")
        
        return {
            'success': True,
            'collaborators_migrated': collaborators_migrated,
            'relationships_migrated': relationships_migrated,
            'verification': verification
        }
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        await database.disconnect()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run migration
    result = asyncio.run(run_migration())
    
    if result['success']:
        print("✅ Migration completed successfully!")
        print(f"📊 Summary: {result}")
    else:
        print("❌ Migration failed!")
        print(f"🚨 Error: {result['error']}")
