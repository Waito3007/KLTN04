# backend/test_multifusion_v2_api.py
"""
Test script for MultiFusion V2 API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_single_commit():
    """Test single commit analysis"""
    url = f"{BASE_URL}/api/repositories/analyze-multifusion-v2-commit"
    
    test_data = {
        "commit_message": "feat: add new user authentication system with JWT tokens",
        "lines_added": 145,
        "lines_removed": 23,
        "files_count": 8,
        "detected_language": "python"
    }
    
    try:
        response = requests.post(url, json=test_data)
        print(f"Single Commit Analysis Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_model_status():
    """Test model status"""
    url = f"{BASE_URL}/api/repositories/1/ai/model-v2-status"
    
    try:
        response = requests.get(url)
        print(f"\nModel Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_batch_analysis():
    """Test batch analysis"""
    url = f"{BASE_URL}/api/repositories/1/ai/batch-analyze-v2"
    
    test_data = {
        "commits": [
            {
                "message": "fix: resolve memory leak in data processing",
                "lines_added": 45,
                "lines_removed": 12,
                "files_count": 3,
                "detected_language": "python"
            },
            {
                "message": "docs: update API documentation",
                "lines_added": 0,
                "lines_removed": 0,
                "files_count": 1,
                "detected_language": "markdown"
            },
            {
                "message": "refactor: optimize database queries",
                "lines_added": 78,
                "lines_removed": 134,
                "files_count": 5,
                "detected_language": "python"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=test_data)
        print(f"\nBatch Analysis Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_model_comparison():
    """Test model comparison"""
    url = f"{BASE_URL}/api/repositories/1/ai/compare-models"
    params = {
        "commit_message": "test: add unit tests for user service",
        "lines_added": 67,
        "lines_removed": 5,
        "files_count": 3,
        "detected_language": "python"
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"\nModel Comparison Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("=== Testing MultiFusion V2 API Endpoints ===")
    
    test_model_status()
    test_single_commit()
    test_batch_analysis()
    test_model_comparison()
    
    print("\n=== Testing completed ===")
