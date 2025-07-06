# backend/demo_member_analysis_v2.py
"""
Demo script showcasing MultiFusion V2 member analysis capabilities
"""

import requests
import json
import time
from typing import List, Dict

BASE_URL = "http://localhost:8000"

def demo_comprehensive_member_analysis():
    """Demo comprehensive member analysis with MultiFusion V2"""
    
    print("🔬 === MultiFusion V2 Member Analysis Demo ===\n")
    
    # Sample member commits data for simulation
    sample_commits = [
        {
            "id": "abc123",
            "message": "feat: implement user authentication with JWT",
            "lines_added": 156,
            "lines_removed": 12,
            "files_count": 8,
            "detected_language": "python",
            "date": "2025-01-01T10:00:00Z"
        },
        {
            "id": "def456", 
            "message": "fix: resolve memory leak in data processing",
            "lines_added": 45,
            "lines_removed": 23,
            "files_count": 3,
            "detected_language": "python",
            "date": "2025-01-02T14:30:00Z"
        },
        {
            "id": "ghi789",
            "message": "docs: update API documentation and examples",
            "lines_added": 0,
            "lines_removed": 0,
            "files_count": 2,
            "detected_language": "markdown",
            "date": "2025-01-03T09:15:00Z"
        },
        {
            "id": "jkl012",
            "message": "refactor: optimize database queries for better performance",
            "lines_added": 78,
            "lines_removed": 134,
            "files_count": 6,
            "detected_language": "python",
            "date": "2025-01-04T16:45:00Z"
        },
        {
            "id": "mno345",
            "message": "test: add comprehensive unit tests for user service",
            "lines_added": 89,
            "lines_removed": 5,
            "files_count": 4,
            "detected_language": "python",
            "date": "2025-01-05T11:20:00Z"
        },
        {
            "id": "pqr678",
            "message": "chore: update dependencies and clean up unused imports",
            "lines_added": 12,
            "lines_removed": 45,
            "files_count": 15,
            "detected_language": "python",
            "date": "2025-01-06T13:10:00Z"
        }
    ]
    
    print("📊 Analyzing sample commits with MultiFusion V2...\n")
    
    # Simulate member analysis by calling the service directly
    try:
        from services.multifusion_v2_service import MultiFusionV2Service
        
        multifusion_v2 = MultiFusionV2Service()
        
        if not multifusion_v2.is_model_available():
            print("❌ MultiFusion V2 model not available")
            return
        
        # Analyze commits
        analysis = multifusion_v2.analyze_member_commits(sample_commits)
        
        print("🎯 === Analysis Results ===")
        print(f"📈 Total Commits Analyzed: {analysis['total_commits']}")
        print(f"🔧 Total Code Changes: {analysis['productivity_metrics']['total_changes']}")
        print(f"📁 Total Files Modified: {analysis['productivity_metrics']['total_files_modified']}")
        print(f"📊 Average Changes per Commit: {analysis['productivity_metrics']['avg_changes_per_commit']}")
        print(f"🗂️ Average Files per Commit: {analysis['productivity_metrics']['avg_files_per_commit']}")
        
        print(f"\n🏆 === Dominant Commit Type ===")
        dominant = analysis['dominant_commit_type']
        print(f"Type: {dominant['type']}")
        print(f"Count: {dominant['count']}")
        print(f"Percentage: {dominant['percentage']}%")
        
        print(f"\n📊 === Commit Type Distribution ===")
        for commit_type, count in analysis['commit_type_distribution'].items():
            percentage = (count / analysis['total_commits']) * 100
            print(f"{commit_type:12}: {count:2} commits ({percentage:5.1f}%)")
        
        print(f"\n💻 === Programming Languages Used ===")
        for lang in analysis['languages_used']:
            print(f"- {lang}")
        
        print(f"\n🔍 === Individual Commit Analysis ===")
        for i, commit in enumerate(analysis['commits'][:3], 1):  # Show first 3
            print(f"\n{i}. Commit: {commit['commit_id'][:8]}...")
            print(f"   Message: {commit['message']}")
            print(f"   Predicted Type: {commit['predicted_type']} (confidence: {commit['confidence']:.3f})")
            print(f"   Changes: +{commit['lines_added']}/-{commit['lines_removed']} lines, {commit['files_count']} files")
            print(f"   Language: {commit['detected_language']}")
        
        if len(analysis['commits']) > 3:
            print(f"\n   ... and {len(analysis['commits']) - 3} more commits")
        
        # Generate insights
        print(f"\n🧠 === AI-Powered Insights ===")
        generate_member_insights(analysis)
        
    except ImportError:
        print("❌ Cannot import MultiFusion V2 service. Testing via API instead...")
        test_via_api(sample_commits)
    except Exception as e:
        print(f"❌ Error in analysis: {e}")

def generate_member_insights(analysis: Dict):
    """Generate insights based on analysis results"""
    
    total_commits = analysis['total_commits']
    commit_types = analysis['commit_type_distribution']
    productivity = analysis['productivity_metrics']
    
    print("🔍 Member Profile Analysis:")
    
    # Developer type analysis
    feat_percentage = (commit_types.get('feat', 0) / total_commits) * 100
    fix_percentage = (commit_types.get('fix', 0) / total_commits) * 100
    refactor_percentage = (commit_types.get('refactor', 0) / total_commits) * 100
    test_percentage = (commit_types.get('test', 0) / total_commits) * 100
    docs_percentage = (commit_types.get('docs', 0) / total_commits) * 100
    
    if feat_percentage >= 40:
        print("🚀 Developer Type: FEATURE BUILDER - Focuses on new functionality")
    elif fix_percentage >= 30:
        print("🔧 Developer Type: BUG HUNTER - Specializes in fixing issues")
    elif refactor_percentage >= 25:
        print("🛠️ Developer Type: CODE OPTIMIZER - Focuses on code improvement")
    elif test_percentage >= 20:
        print("✅ Developer Type: QUALITY ASSURER - Emphasizes testing")
    elif docs_percentage >= 15:
        print("📚 Developer Type: DOCUMENTATION CHAMPION - Focuses on documentation")
    else:
        print("🎯 Developer Type: BALANCED CONTRIBUTOR - Well-rounded development")
    
    # Productivity insights
    avg_changes = productivity['avg_changes_per_commit']
    if avg_changes > 100:
        print("📊 Commit Style: LARGE COMMITS - Prefers substantial changes")
    elif avg_changes > 50:
        print("📊 Commit Style: MEDIUM COMMITS - Balanced approach")
    else:
        print("📊 Commit Style: SMALL COMMITS - Incremental development")
    
    # Quality insights
    total_changes = productivity['total_changes']
    if total_changes > 500:
        print("💪 Productivity Level: HIGH - Significant code contribution")
    elif total_changes > 200:
        print("📈 Productivity Level: MEDIUM - Steady contribution")
    else:
        print("🌱 Productivity Level: GROWING - Building up contribution")
    
    # Language expertise
    languages = analysis['languages_used']
    if len(languages) > 3:
        print("🌐 Language Skills: POLYGLOT - Works with multiple languages")
    elif len(languages) == 1:
        print(f"🎯 Language Skills: SPECIALIST - Focused on {languages[0]}")
    else:
        print("⚖️ Language Skills: FOCUSED - Works with select languages")

def test_via_api(commits: List[Dict]):
    """Test member analysis via API call"""
    
    print("Testing via API...")
    url = f"{BASE_URL}/api/repositories/1/ai/batch-analyze-v2"
    
    # Convert commits to API format
    api_commits = []
    for commit in commits:
        api_commits.append({
            "message": commit["message"],
            "lines_added": commit["lines_added"],
            "lines_removed": commit["lines_removed"],
            "files_count": commit["files_count"],
            "detected_language": commit["detected_language"]
        })
    
    try:
        response = requests.post(url, json={"commits": api_commits})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Test Successful - Analyzed {result['total_analyzed']} commits")
        else:
            print(f"❌ API Test Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API Request Failed: {e}")

if __name__ == "__main__":
    demo_comprehensive_member_analysis()
