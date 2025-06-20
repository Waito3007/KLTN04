# backend/test_refined_commit_endpoints.py
"""
Test script for refined commit endpoints after cleanup and new additions

This script tests:
1. Database endpoints (fast, stored data)
2. GitHub direct fetch endpoints (real-time data) 
3. Sync endpoints
4. Analytics endpoints

Updated endpoint structure:
- Removed redundant /github/{owner}/{repo}/commits (old one)
- Removed redundant /commits/{owner}/{repo}/commits (duplicated logic)
- Added /github/{owner}/{repo}/branches/{branch_name}/commits (GitHub direct fetch)
- Added /github/{owner}/{repo}/commits (GitHub direct fetch with full filters)
- Consolidated /commits/{owner}/{repo}/commits (database only)
"""

import asyncio
import httpx
import json
from typing import Dict, List

# Test configuration
BASE_URL = "http://localhost:8000/api"
GITHUB_TOKEN = "your_github_token_here"  # Replace with actual token
TEST_REPO = {
    "owner": "Waito3007",
    "repo": "KLTN04",
    "branch": "main"
}

class CommitEndpointTester:
    def __init__(self, base_url: str, github_token: str = None):
        self.base_url = base_url
        self.github_token = github_token
        self.headers = {}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
    
    async def test_endpoint(self, endpoint: str, method: str = "GET", params: Dict = None, 
                          expect_auth: bool = False, description: str = ""):
        """Test a single endpoint"""
        print(f"\n{'='*60}")
        print(f"Testing: {method} {endpoint}")
        print(f"Description: {description}")
        print(f"Expected auth: {expect_auth}")
        
        if params:
            print(f"Parameters: {json.dumps(params, indent=2)}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = self.headers if expect_auth else {}
                
                if method == "GET":
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers,
                        params=params or {}
                    )
                elif method == "POST":
                    response = await client.post(
                        f"{self.base_url}{endpoint}",
                        headers=headers,
                        params=params or {}
                    )
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ SUCCESS")
                    
                    # Print useful summary info
                    if isinstance(data, dict):
                        if "commits" in data:
                            print(f"   - Commits returned: {len(data['commits'])}")
                        if "branches" in data:
                            print(f"   - Branches returned: {len(data['branches'])}")
                        if "repository" in data:
                            print(f"   - Repository: {data['repository']}")
                        if "source" in data:
                            print(f"   - Data source: {data['source']}")
                        if "total_commits_saved" in data:
                            print(f"   - Total commits saved: {data['total_commits_saved']}")
                    
                    # Show first item if it's a list
                    if isinstance(data, dict) and "commits" in data and data["commits"]:
                        first_commit = data["commits"][0]
                        print(f"   - First commit SHA: {first_commit.get('sha', 'N/A')}")
                        print(f"   - First commit message: {first_commit.get('message', 'N/A')[:60]}...")
                
                elif response.status_code == 401:
                    print("❌ AUTHENTICATION REQUIRED")
                    if not expect_auth:
                        print("   - This endpoint requires GitHub token")
                
                elif response.status_code == 404:
                    print("❌ NOT FOUND")
                    print(f"   - {response.text}")
                
                elif response.status_code == 429:
                    print("⚠️ RATE LIMITED")
                    print("   - GitHub API rate limit exceeded")
                
                else:
                    print(f"❌ ERROR: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   - Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"   - Raw response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
    
    async def run_all_tests(self):
        """Run comprehensive tests on all commit endpoints"""
        print("🚀 Starting Commit Endpoints Test Suite")
        print(f"Base URL: {self.base_url}")
        print(f"Test Repository: {TEST_REPO['owner']}/{TEST_REPO['repo']}")
        print(f"GitHub Token: {'✅ Provided' if self.github_token else '❌ Not provided'}")
        
        # 1. DATABASE ENDPOINTS (Fast, stored data)
        print(f"\n{'#'*60}")
        print("# 1. DATABASE ENDPOINTS (Fast, stored data)")
        print(f"{'#'*60}")
        
        await self.test_endpoint(
            f"/commits/{TEST_REPO['owner']}/{TEST_REPO['repo']}/branches/{TEST_REPO['branch']}/commits",
            description="Get commits by branch from database",
            params={"limit": 5}
        )
        
        await self.test_endpoint(
            f"/commits/{TEST_REPO['owner']}/{TEST_REPO['repo']}/commits",
            description="Get all repo commits from database",
            params={"limit": 5}
        )
        
        await self.test_endpoint(
            f"/commits/{TEST_REPO['owner']}/{TEST_REPO['repo']}/branches",
            description="Get all branches with commit stats"
        )
        
        await self.test_endpoint(
            f"/commits/{TEST_REPO['owner']}/{TEST_REPO['repo']}/compare/main...dev",
            description="Compare commits between branches"
        )
        
        # 2. GITHUB DIRECT FETCH ENDPOINTS (Real-time, live data)
        print(f"\n{'#'*60}")
        print("# 2. GITHUB DIRECT FETCH ENDPOINTS (Real-time, live data)")
        print(f"{'#'*60}")
        
        await self.test_endpoint(
            f"/github/{TEST_REPO['owner']}/{TEST_REPO['repo']}/branches/{TEST_REPO['branch']}/commits",
            description="Fetch branch commits directly from GitHub",
            params={"per_page": 5},
            expect_auth=True
        )
        
        await self.test_endpoint(
            f"/github/{TEST_REPO['owner']}/{TEST_REPO['repo']}/commits",
            description="Fetch repo commits directly from GitHub with filters",
            params={"per_page": 5, "sha": TEST_REPO['branch']},
            expect_auth=True
        )
        
        # 3. SYNC & MANAGEMENT ENDPOINTS
        print(f"\n{'#'*60}")
        print("# 3. SYNC & MANAGEMENT ENDPOINTS")
        print(f"{'#'*60}")
        
        # Note: These are POST endpoints that modify data, so we'll just test the structure
        await self.test_endpoint(
            f"/github/{TEST_REPO['owner']}/{TEST_REPO['repo']}/sync-commits",
            method="POST",
            description="Sync commits from GitHub to database",
            params={"branch": TEST_REPO['branch'], "max_pages": 1},
            expect_auth=True
        )
        
        await self.test_endpoint(
            f"/commits/{TEST_REPO['owner']}/{TEST_REPO['repo']}/validate-commit-consistency",
            method="POST",
            description="Validate and fix commit-branch consistency"
        )
        
        # 4. ANALYTICS & STATS ENDPOINTS
        print(f"\n{'#'*60}")
        print("# 4. ANALYTICS & STATS ENDPOINTS")
        print(f"{'#'*60}")
        
        await self.test_endpoint(
            f"/github/{TEST_REPO['owner']}/{TEST_REPO['repo']}/commit-stats",
            description="Get comprehensive commit statistics"
        )
        
        # 5. SPECIFIC COMMIT DETAILS
        print(f"\n{'#'*60}")
        print("# 5. SPECIFIC COMMIT DETAILS")
        print(f"{'#'*60}")
        
        # Test with a known commit SHA (we'll use a placeholder)
        test_sha = "a1b2c3d4e5f6"  # Replace with actual SHA for real testing
        await self.test_endpoint(
            f"/commits/{test_sha}",
            description="Get specific commit details by SHA"
        )
        
        print(f"\n{'='*60}")
        print("🏁 Test Suite Completed!")
        print("📋 Summary:")
        print("   - Database endpoints: Fast access to stored data")
        print("   - GitHub direct endpoints: Real-time data (requires auth)")
        print("   - Sync endpoints: Populate database from GitHub")
        print("   - Analytics endpoints: Statistics and insights")
        print("   - Individual commit lookup: Detailed commit info")


async def main():
    """Run the test suite"""
    tester = CommitEndpointTester(BASE_URL, GITHUB_TOKEN)
    await tester.run_all_tests()


if __name__ == "__main__":
    print("Commit Endpoints Test Suite")
    print("=" * 60)
    
    if GITHUB_TOKEN == "your_github_token_here":
        print("⚠️  WARNING: Please update GITHUB_TOKEN in the script for full testing")
        print("   GitHub-dependent endpoints will show authentication errors")
    
    asyncio.run(main())
