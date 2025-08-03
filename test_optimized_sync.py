#!/usr/bin/env python3
"""
Test script để kiểm tra performance của optimized sync function
"""
import asyncio
import time
import httpx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_OWNER = "microsoft"
TEST_REPO = "vscode"  # Large repo for performance testing
# Or use a smaller repo for initial testing
SMALL_TEST_OWNER = "octocat"
SMALL_TEST_REPO = "Hello-World"

# GitHub token (you need to provide this)
GITHUB_TOKEN = "your_github_token_here"  # Replace with actual token

async def test_github_api_performance():
    """Test basic GitHub API performance"""
    print("🧪 Testing basic GitHub API performance...")
    
    async with httpx.AsyncClient() as client:
        # Test repository info fetch
        start_time = time.time()
        
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        response = await client.get(
            f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}",
            headers=headers
        )
        
        if response.status_code == 200:
            repo_data = response.json()
            fetch_time = time.time() - start_time
            print(f"✅ Repository info fetched in {fetch_time:.2f}s")
            print(f"📊 Repo: {repo_data['full_name']}, Stars: {repo_data['stargazers_count']}")
        else:
            print(f"❌ Failed to fetch repo info: {response.status_code}")
            return False
    
    return True

async def test_concurrent_requests():
    """Test concurrent API requests with semaphore"""
    print("\n🧪 Testing concurrent requests...")
    
    # Create semaphore
    semaphore = asyncio.Semaphore(10)
    
    async def fetch_with_semaphore(url, client, headers):
        async with semaphore:
            response = await client.get(url, headers=headers)
            return response.status_code, len(response.content) if response.status_code == 200 else 0
    
    # Test URLs
    test_urls = [
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/branches",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/commits?per_page=10",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/issues?state=all&per_page=10",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/pulls?state=all&per_page=10"
    ]
    
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        # Execute all requests concurrently
        tasks = [fetch_with_semaphore(url, client, headers) for url in test_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        concurrent_time = time.time() - start_time
        
        print(f"✅ {len(test_urls)} concurrent requests completed in {concurrent_time:.2f}s")
        
        success_count = sum(1 for status, _ in results if isinstance(status, int) and status == 200)
        print(f"📊 Successful requests: {success_count}/{len(test_urls)}")
    
    return concurrent_time

async def test_sequential_requests():
    """Test sequential API requests for comparison"""
    print("\n🧪 Testing sequential requests...")
    
    test_urls = [
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/branches",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/commits?per_page=10",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/issues?state=all&per_page=10",
        f"https://api.github.com/repos/{SMALL_TEST_OWNER}/{SMALL_TEST_REPO}/pulls?state=all&per_page=10"
    ]
    
    start_time = time.time()
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        results = []
        for url in test_urls:
            response = await client.get(url, headers=headers)
            results.append((response.status_code, len(response.content) if response.status_code == 200 else 0))
        
        sequential_time = time.time() - start_time
        
        print(f"✅ {len(test_urls)} sequential requests completed in {sequential_time:.2f}s")
        
        success_count = sum(1 for status, _ in results if status == 200)
        print(f"📊 Successful requests: {success_count}/{len(test_urls)}")
    
    return sequential_time

async def main():
    """Main test function"""
    print("🚀 Starting optimized sync performance tests")
    print(f"🎯 Testing with repository: {SMALL_TEST_OWNER}/{SMALL_TEST_REPO}")
    
    if GITHUB_TOKEN == "your_github_token_here":
        print("❌ Please set your GitHub token in the script")
        return
    
    # Test 1: Basic API performance
    basic_success = await test_github_api_performance()
    if not basic_success:
        print("❌ Basic API test failed")
        return
    
    # Test 2: Sequential vs Concurrent performance
    sequential_time = await test_sequential_requests()
    concurrent_time = await test_concurrent_requests()
    
    # Calculate performance improvement
    improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
    
    print(f"\n📈 Performance Comparison:")
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Improvement: {improvement:.1f}% faster")
    
    print("\n✅ Performance tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
