#!/usr/bin/env python3
"""
Script để fix commit-branch consistency issues
Chạy script này để kiểm tra và sửa các vấn đề inconsistency trong database
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import database
from services.commit_service import validate_and_fix_commit_branch_consistency, get_repo_id_by_owner_and_name
from db.models.repositories import repositories
from sqlalchemy import select
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fix_all_repositories():
    """Fix commit-branch consistency for all repositories"""
    try:
        await database.connect()
        
        # Get all repositories
        query = select(repositories.c.id, repositories.c.owner, repositories.c.name)
        repos = await database.fetch_all(query)
        
        logger.info(f"🔍 Found {len(repos)} repositories to check")
        
        total_fixed = 0
        
        for repo in repos:
            repo_name = f"{repo.owner}/{repo.name}"
            logger.info(f"📝 Checking {repo_name}...")
            
            try:
                fixed_count = await validate_and_fix_commit_branch_consistency(repo.id)
                total_fixed += fixed_count
                
                if fixed_count > 0:
                    logger.info(f"✅ Fixed {fixed_count} inconsistencies in {repo_name}")
                else:
                    logger.info(f"✅ No issues found in {repo_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error checking {repo_name}: {e}")
        
        logger.info(f"🎯 SUMMARY: Fixed {total_fixed} total inconsistencies across {len(repos)} repositories")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await database.disconnect()

async def fix_specific_repository(owner: str, name: str):
    """Fix commit-branch consistency for a specific repository"""
    try:
        await database.connect()
        
        repo_id = await get_repo_id_by_owner_and_name(owner, name)
        if not repo_id:
            logger.error(f"❌ Repository {owner}/{name} not found")
            return
        
        logger.info(f"🔍 Checking repository {owner}/{name}...")
        
        fixed_count = await validate_and_fix_commit_branch_consistency(repo_id)
        
        if fixed_count > 0:
            logger.info(f"✅ Fixed {fixed_count} inconsistencies in {owner}/{name}")
        else:
            logger.info(f"✅ No issues found in {owner}/{name}")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        await database.disconnect()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Fix specific repository
        owner, name = sys.argv[1], sys.argv[2]
        asyncio.run(fix_specific_repository(owner, name))
    else:
        # Fix all repositories
        print("🚀 Fixing commit-branch consistency for all repositories...")
        print("📝 To fix a specific repository, run: python fix_commit_branch_consistency.py OWNER REPO_NAME")
        asyncio.run(fix_all_repositories())
