#!/usr/bin/env python3
"""
Test script để verify các endpoints commit by branch mới
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api"

async def test_commit_endpoints():
    """Test các endpoints commit mới"""
    
    # Test data - thay đổi theo repository thực tế trong database
    test_repos = [
        {"owner": "microsoft", "repo": "vscode"},
        {"owner": "facebook", "repo": "react"},
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for repo_info in test_repos:
            owner = repo_info["owner"]
            repo = repo_info["repo"]
            
            print(f"\n🔍 Testing repository: {owner}/{repo}")
            print("=" * 50)
            
            try:
                # 1. Test get branches with commit stats
                print(f"📋 1. Getting branches for {owner}/{repo}...")
                response = await client.get(f"{BASE_URL}/commits/{owner}/{repo}/branches")
                
                if response.status_code == 200:
                    data = response.json()
                    branches = data.get("branches", [])
                    print(f"✅ Found {len(branches)} branches")
                    
                    # Show first few branches
                    for i, branch in enumerate(branches[:3]):
                        print(f"   - {branch['name']}: {branch['actual_commit_count']} commits (default: {branch['is_default']})")
                    
                    # Test branch-specific commits for each branch
                    for branch in branches[:2]:  # Test first 2 branches
                        branch_name = branch["name"]
                        
                        print(f"\n📝 2. Getting commits for branch '{branch_name}'...")
                        commit_response = await client.get(
                            f"{BASE_URL}/commits/{owner}/{repo}/branches/{branch_name}/commits",
                            params={"limit": 10}
                        )
                        
                        if commit_response.status_code == 200:
                            commit_data = commit_response.json()
                            commits = commit_data.get("commits", [])
                            print(f"✅ Found {len(commits)} commits in branch '{branch_name}'")
                            
                            # Show latest commit
                            if commits:
                                latest = commits[0]
                                print(f"   Latest: {latest['sha'][:8]} - {latest['message'][:50]}...")
                                print(f"   Author: {latest['author_name']} ({latest['date'][:10]})")
                        else:
                            print(f"❌ Failed to get commits for branch '{branch_name}': {commit_response.status_code}")
                    
                    # 3. Test branch comparison if we have multiple branches
                    if len(branches) >= 2:
                        base_branch = branches[0]["name"]
                        head_branch = branches[1]["name"]
                        
                        print(f"\n🔄 3. Comparing branches {base_branch}...{head_branch}...")
                        compare_response = await client.get(
                            f"{BASE_URL}/commits/{owner}/{repo}/compare/{base_branch}...{head_branch}",
                            params={"limit": 5}
                        )
                        
                        if compare_response.status_code == 200:
                            compare_data = compare_response.json()
                            ahead_commits = compare_data.get("commits_ahead", [])
                            print(f"✅ {head_branch} is {len(ahead_commits)} commits ahead of {base_branch}")
                            
                            for commit in ahead_commits[:3]:
                                print(f"   + {commit['sha'][:8]} - {commit['message'][:40]}...")
                        else:
                            print(f"❌ Failed to compare branches: {compare_response.status_code}")
                    
                    # 4. Test enhanced commits endpoint with branch filter
                    if branches:
                        test_branch = branches[0]["name"]
                        print(f"\n🔍 4. Testing enhanced commits endpoint with branch filter...")
                        enhanced_response = await client.get(
                            f"{BASE_URL}/commits/{owner}/{repo}/commits",
                            params={"branch": test_branch, "limit": 5}
                        )
                        
                        if enhanced_response.status_code == 200:
                            enhanced_data = enhanced_response.json()
                            commits = enhanced_data.get("commits", [])
                            print(f"✅ Enhanced endpoint returned {len(commits)} commits for branch '{test_branch}'")
                        else:
                            print(f"❌ Enhanced endpoint failed: {enhanced_response.status_code}")
                    
                    # 5. Test consistency validation
                    print(f"\n🔧 5. Testing commit-branch consistency validation...")
                    validation_response = await client.post(
                        f"{BASE_URL}/commits/{owner}/{repo}/validate-commit-consistency"
                    )
                    
                    if validation_response.status_code == 200:
                        validation_data = validation_response.json()
                        fixed_count = validation_data.get("inconsistencies_fixed", 0)
                        print(f"✅ Consistency validation completed. Fixed {fixed_count} inconsistencies.")
                    else:
                        print(f"❌ Consistency validation failed: {validation_response.status_code}")
                
                else:
                    print(f"❌ Failed to get branches: {response.status_code}")
                    if response.status_code == 404:
                        print(f"   Repository {owner}/{repo} not found in database")
                    
            except Exception as e:
                print(f"❌ Error testing {owner}/{repo}: {e}")
    
    print(f"\n🎯 Testing completed at {datetime.now()}")

if __name__ == "__main__":
    print("🚀 Testing commit-by-branch endpoints...")
    print("📝 Make sure the backend server is running on http://localhost:8000")
    print("💾 Make sure you have some repositories synced in the database")
    
    asyncio.run(test_commit_endpoints())
