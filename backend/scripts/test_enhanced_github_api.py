# backend/scripts/test_enhanced_github_api.py
"""
Script to test enhanced GitHub API integration for fetching detailed commit metadata
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.github_service import (
    fetch_commit_details,
    fetch_enhanced_commits_batch,
    fetch_commit_files_metadata,
    get_commit_comparison
)

async def test_enhanced_commit_details():
    """Test fetching enhanced commit details"""
    print("=== Testing Enhanced Commit Details ===")
    
    # Test with a popular repository
    owner = "facebook"
    repo = "react"
    
    try:
        # Get recent commits first
        from services.github_service import fetch_commits
        commits = await fetch_commits(
            token=os.getenv("GITHUB_TOKEN"),
            owner=owner,
            name=repo,
            branch="main",
            per_page=5
        )
        
        if commits:
            commit_sha = commits[0]["sha"]
            print(f"Testing with commit: {commit_sha}")
            
            # Fetch enhanced details
            enhanced_commit = await fetch_commit_details(commit_sha, owner, repo)
            
            if enhanced_commit:
                print(f"✅ Successfully fetched enhanced commit details")
                print(f"   - SHA: {enhanced_commit['sha']}")
                print(f"   - Files changed: {enhanced_commit['files_changed']}")
                print(f"   - Additions: {enhanced_commit['additions']}")
                print(f"   - Deletions: {enhanced_commit['deletions']}")
                print(f"   - Total changes: {enhanced_commit['total_changes']}")
                print(f"   - Is merge: {enhanced_commit['is_merge']}")
                print(f"   - Modified files: {len(enhanced_commit['modified_files'])}")
                print(f"   - File types: {enhanced_commit['file_types']}")
                print(f"   - Modified directories: {enhanced_commit['modified_directories']}")
                return True
            else:
                print("❌ Failed to fetch enhanced commit details")
                return False
        else:
            print("❌ No commits found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced commit details: {e}")
        return False

async def test_enhanced_commits_batch():
    """Test fetching enhanced commits in batch"""
    print("\n=== Testing Enhanced Commits Batch ===")
    
    owner = "torvalds"
    repo = "linux"
    
    try:
        # Fetch last 10 commits with enhanced metadata
        enhanced_commits = await fetch_enhanced_commits_batch(
            owner=owner,
            repo=repo,
            branch="master",
            max_commits=10
        )
        
        if enhanced_commits:
            print(f"✅ Successfully fetched {len(enhanced_commits)} enhanced commits")
            
            total_files = sum(c['files_changed'] for c in enhanced_commits)
            total_additions = sum(c['additions'] for c in enhanced_commits)
            total_deletions = sum(c['deletions'] for c in enhanced_commits)
            merge_commits = sum(1 for c in enhanced_commits if c['is_merge'])
            
            print(f"   - Total files changed: {total_files}")
            print(f"   - Total additions: {total_additions}")
            print(f"   - Total deletions: {total_deletions}")
            print(f"   - Merge commits: {merge_commits}")
            
            # Show file types distribution
            all_file_types = {}
            for commit in enhanced_commits:
                for file_type, count in commit['file_types'].items():
                    all_file_types[file_type] = all_file_types.get(file_type, 0) + count
            
            print(f"   - File types: {dict(list(all_file_types.items())[:5])}")
            return True
        else:
            print("❌ No enhanced commits fetched")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced commits batch: {e}")
        return False

async def test_commit_files_metadata():
    """Test fetching detailed file metadata"""
    print("\n=== Testing Commit Files Metadata ===")
    
    owner = "microsoft"
    repo = "vscode"
    
    try:
        # Get a recent commit
        from services.github_service import fetch_commits
        commits = await fetch_commits(
            token=os.getenv("GITHUB_TOKEN"),
            owner=owner,
            name=repo,
            branch="main",
            per_page=3
        )
        
        if commits:
            commit_sha = commits[0]["sha"]
            print(f"Testing file metadata for commit: {commit_sha}")
            
            file_metadata = await fetch_commit_files_metadata(owner, repo, commit_sha)
            
            if file_metadata:
                print(f"✅ Successfully fetched file metadata")
                print(f"   - Files changed: {file_metadata['files_changed']}")
                print(f"   - Total additions: {file_metadata['size_changes']['additions']}")
                print(f"   - Total deletions: {file_metadata['size_changes']['deletions']}")
                print(f"   - File categories: {file_metadata['file_categories']}")
                print(f"   - File types: {dict(list(file_metadata['file_types'].items())[:3])}")
                print(f"   - Directories: {dict(list(file_metadata['modified_directories'].items())[:3])}")
                return True
            else:
                print("❌ Failed to fetch file metadata")
                return False
        else:
            print("❌ No commits found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing commit files metadata: {e}")
        return False

async def test_commit_comparison():
    """Test commit comparison"""
    print("\n=== Testing Commit Comparison ===")
    
    owner = "python"
    repo = "cpython"
    
    try:
        # Get two recent commits
        from services.github_service import fetch_commits
        commits = await fetch_commits(
            token=os.getenv("GITHUB_TOKEN"),
            owner=owner,
            name=repo,
            branch="main",
            per_page=5
        )
        
        if len(commits) >= 2:
            base_sha = commits[1]["sha"]
            head_sha = commits[0]["sha"]
            
            print(f"Comparing {base_sha[:8]}...{head_sha[:8]}")
            
            comparison = await get_commit_comparison(owner, repo, base_sha, head_sha)
            
            if comparison:
                print(f"✅ Successfully compared commits")
                print(f"   - Status: {comparison['status']}")
                print(f"   - Total commits: {comparison['total_commits']}")
                print(f"   - Files changed: {len(comparison.get('files', []))}")
                print(f"   - Additions: {comparison['stats']['additions']}")
                print(f"   - Deletions: {comparison['stats']['deletions']}")
                print(f"   - Total changes: {comparison['stats']['total']}")
                return True
            else:
                print("❌ Failed to compare commits")
                return False
        else:
            print("❌ Not enough commits for comparison")
            return False
            
    except Exception as e:
        print(f"❌ Error testing commit comparison: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Enhanced GitHub API Integration")
    print("=" * 50)
    
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️  Warning: GITHUB_TOKEN not set, some tests may fail")
    
    results = []
    
    # Run all tests
    test_functions = [
        test_enhanced_commit_details,
        test_enhanced_commits_batch,
        test_commit_files_metadata,
        test_commit_comparison
    ]
    
    for test_func in test_functions:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced GitHub API integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
