# backend/test_multifusion_integration.py
"""
Test MultiFusion V2 integration với backend API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_multifusion_v2_commit_analysis():
    """Test MultiFusion V2 commit analysis endpoint"""
    print("🧪 Testing MultiFusion V2 Commit Analysis API")
    print("=" * 50)
    
    # Test data
    test_commits = [
        {
            "commit_message": "fix: resolve authentication bug in login system",
            "lines_added": "15",
            "lines_removed": "8",
            "file_count": "3",
            "diff_content": "- if (user.password) {\n+ if (user.password && user.password.length > 0) {"
        },
        {
            "commit_message": "feat: add new dashboard component with charts",
            "lines_added": "150",
            "lines_removed": "5",
            "file_count": "6"
        },
        {
            "commit_message": "docs: update README with installation instructions",
            "lines_added": "25",
            "lines_removed": "2",
            "file_count": "1"
        },
        {
            "commit_message": "test: add unit tests for user service",
            "lines_added": "80",
            "lines_removed": "0",
            "file_count": "2"
        }
    ]
    
    try:
        for i, commit_data in enumerate(test_commits, 1):
            print(f"\n🎯 Test {i}: {commit_data['commit_message'][:50]}...")
            
            response = requests.post(
                f"{BASE_URL}/api/v1/member-analysis/analyze-multifusion-v2-commit",
                json=commit_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ Status: {response.status_code}")
                print(f"    📊 Predicted: {result.get('predicted_type', 'unknown')}")
                print(f"    🎯 Confidence: {result.get('confidence', 0):.4f}")
                
                if result.get('all_probabilities'):
                    top_3 = sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
                    print(f"    📈 Top 3: {top_3}")
            else:
                print(f"    ❌ Error {response.status_code}: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_service_health():
    """Test if backend service is healthy"""
    print("\n🏥 Testing Backend Health")
    print("=" * 30)
    
    try:
        # Test basic health
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Backend is healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except:
        print("❌ Backend is not responding")
        return False

if __name__ == "__main__":
    print("🚀 MultiFusion V2 Integration Test")
    print("=" * 60)
    
    # Test backend health first
    if test_service_health():
        test_multifusion_v2_commit_analysis()
    else:
        print("\n💡 Please start the backend server first:")
        print("   cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    
    print("\n✅ Testing completed!")
