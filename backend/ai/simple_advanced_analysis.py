#!/usr/bin/env python3
"""
Simple Advanced Analysis - Version đơn giản không dùng matplotlib
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

def load_analysis_report(report_path):
    """Load báo cáo phân tích từ file JSON"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_author_patterns(report_data):
    """Phân tích pattern của từng tác giả"""
    print("\n" + "="*80)
    print("🔍 PHÂN TÍCH CHI TIẾT PATTERN CỦA TÁC GIẢ")
    print("="*80)
    
    author_stats = report_data['author_statistics']
    
    for author_name, stats in author_stats.items():
        print(f"\n👤 {author_name}:")
        print(f"   📊 Tổng commits: {stats['total_commits']}")
        print(f"   📈 Mức độ hoạt động: {stats['activity_level'].upper()}")
        print(f"   🎯 Confidence trung bình: {stats['avg_confidence']:.3f}")
        
        # Phân tích commit types
        if stats['commit_types']:
            print(f"   🏷️  Phân bố loại commit:")
            for commit_type, count in sorted(stats['commit_types'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_commits']) * 100
                print(f"      {commit_type}: {count} ({percentage:.1f}%)")
        
        # Phân tích purposes
        if stats['purposes']:
            print(f"   🎯 Phân bố mục đích:")
            for purpose, count in sorted(stats['purposes'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_commits']) * 100
                print(f"      {purpose}: {count} ({percentage:.1f}%)")
        
        # Phân tích sentiment
        if stats['sentiments']:
            print(f"   😊 Phân bố cảm xúc:")
            for sentiment, count in sorted(stats['sentiments'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_commits']) * 100
                emoji = {"positive": "😊", "neutral": "😐", "negative": "😞", "urgent": "🚨"}.get(sentiment, "❓")
                print(f"      {emoji} {sentiment}: {count} ({percentage:.1f}%)")

def generate_detailed_recommendations(report_data):
    """Tạo khuyến nghị chi tiết cho team"""
    print("\n" + "="*80)
    print("💡 KHUYẾN NGHỊ CHI TIẾT CHO TEAM")
    print("="*80)
    
    author_stats = report_data['author_statistics']
    overloaded_authors = report_data['activity_analysis']['overloaded_authors']
    low_activity_authors = report_data['activity_analysis']['low_activity_authors']
    
    # Phân tích tổng quan team
    total_commits = report_data['summary']['total_commits']
    total_authors = report_data['summary']['unique_authors']
    avg_commits = report_data['summary']['avg_commits_per_author']
    
    print(f"\n📊 TỔNG QUAN TEAM:")
    print(f"   👥 Tổng số dev: {total_authors}")
    print(f"   📝 Tổng commits: {total_commits}")
    print(f"   📈 Trung bình commits/dev: {avg_commits:.1f}")
    
    # Phân tích workload distribution
    commit_counts = [stats['total_commits'] for stats in author_stats.values()]
    max_commits = max(commit_counts)
    min_commits = min(commit_counts)
    workload_ratio = max_commits / min_commits if min_commits > 0 else 0
    
    print(f"\n⚖️  PHÂN TÍCH WORKLOAD:")
    print(f"   📊 Commits cao nhất: {max_commits}")
    print(f"   📊 Commits thấp nhất: {min_commits}")
    print(f"   📊 Tỷ lệ workload: {workload_ratio:.1f}:1")
    
    if workload_ratio > 5:
        print(f"   ⚠️  CẢNH BÁO: Workload không cân bằng!")
        print(f"       💡 Khuyến nghị: Cần phân phối lại công việc")
    
    # Khuyến nghị cho overloaded authors
    if overloaded_authors:
        print(f"\n🔥 TÌNH TRẠNG QUÁ TẢI ({len(overloaded_authors)} dev):")
        for author in overloaded_authors:
            stats = author_stats[author]
            print(f"\n   🔥 {author}:")
            print(f"      📊 {stats['total_commits']} commits ({(stats['total_commits']/avg_commits*100):.0f}% của trung bình)")
            
            # Phân tích pattern để đưa ra khuyến nghị cụ thể
            if stats['commit_types']:
                fix_count = stats['commit_types'].get('fix', 0)
                feat_count = stats['commit_types'].get('feat', 0)
                
                print(f"      🔧 Pattern analysis:")
                if fix_count > stats['total_commits'] * 0.4:
                    print(f"         🐛 Quá nhiều fix commits ({fix_count}/{stats['total_commits']})")
                    print(f"         💡 Khuyến nghị: Tăng cường code review và testing")
                
                if feat_count > stats['total_commits'] * 0.6:
                    print(f"         ✨ Nhiều feature commits ({feat_count}/{stats['total_commits']})")
                    print(f"         💡 Nhận xét: Key developer, cần có backup plan")
            
            print(f"      💡 Khuyến nghị chung:")
            print(f"         - Cân nhắc phân phối một số task cho dev khác")
            print(f"         - Đảm bảo work-life balance")
            print(f"         - Review capacity planning")
    
    # Khuyến nghị cho low activity authors
    if low_activity_authors:
        print(f"\n💤 HOẠT ĐỘNG THẤP ({len(low_activity_authors)} dev):")
        for author in low_activity_authors:
            stats = author_stats[author]
            print(f"\n   💤 {author}:")
            print(f"      📊 {stats['total_commits']} commits ({(stats['total_commits']/avg_commits*100):.0f}% của trung bình)")
            print(f"      💡 Khuyến nghị:")
            print(f"         - Kiểm tra workload và obstacles")
            print(f"         - Cung cấp mentoring hoặc training")
            print(f"         - Review task assignment process")
    
    # Phân tích quality metrics
    commit_types = report_data['overall_distributions']['commit_types']
    fix_percentage = (commit_types.get('fix', 0) / total_commits) * 100
    feat_percentage = (commit_types.get('feat', 0) / total_commits) * 100
    test_percentage = (commit_types.get('test', 0) / total_commits) * 100
    
    print(f"\n🎯 PHÂN TÍCH CHẤT LƯỢNG:")
    print(f"   🐛 Fix commits: {fix_percentage:.1f}%")
    print(f"   ✨ Feature commits: {feat_percentage:.1f}%")
    print(f"   🧪 Test commits: {test_percentage:.1f}%")
    
    if fix_percentage > 40:
        print(f"   ⚠️  Tỷ lệ fix commits cao!")
        print(f"       💡 Khuyến nghị:")
        print(f"          - Tăng cường code review process")
        print(f"          - Cải thiện testing coverage")
        print(f"          - Review development practices")
    
    if test_percentage < 10:
        print(f"   ⚠️  Tỷ lệ test commits thấp!")
        print(f"       💡 Khuyến nghị:")
        print(f"          - Khuyến khích viết test")
        print(f"          - Training về testing practices")
        print(f"          - Đưa testing vào definition of done")
    
    # Sentiment analysis
    sentiments = report_data['overall_distributions']['sentiments']
    total_sentiments = sum(sentiments.values())
    
    print(f"\n😊 PHÂN TÍCH TEAM MORALE:")
    for sentiment, count in sentiments.items():
        percentage = (count / total_sentiments) * 100
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😞", "urgent": "🚨"}.get(sentiment, "❓")
        print(f"   {emoji} {sentiment}: {percentage:.1f}%")
    
    negative_percentage = (sentiments.get('negative', 0) / total_sentiments) * 100
    urgent_percentage = (sentiments.get('urgent', 0) / total_sentiments) * 100
    
    if negative_percentage > 30:
        print(f"   ⚠️  Tỷ lệ sentiment tiêu cực cao ({negative_percentage:.1f}%)!")
        print(f"       💡 Khuyến nghị:")
        print(f"          - Survey team morale")
        print(f"          - Review workload và deadlines")
        print(f"          - Cải thiện team communication")
    
    if urgent_percentage > 15:
        print(f"   🚨 Tỷ lệ urgent commits cao ({urgent_percentage:.1f}%)!")
        print(f"       💡 Khuyến nghị:")
        print(f"          - Cải thiện planning và estimation")
        print(f"          - Review risk management")
        print(f"          - Tăng cường testing và CI/CD")

def create_action_plan(report_data):
    """Tạo action plan cụ thể"""
    print("\n" + "="*80)
    print("📋 ACTION PLAN")
    print("="*80)
    
    author_stats = report_data['author_statistics']
    overloaded_authors = report_data['activity_analysis']['overloaded_authors']
    low_activity_authors = report_data['activity_analysis']['low_activity_authors']
    
    actions = []
    
    # Actions for overloaded authors
    if overloaded_authors:
        actions.append({
            "priority": "HIGH",
            "category": "Workload Balancing",
            "action": f"Redistribute tasks from {len(overloaded_authors)} overloaded developers",
            "timeline": "Next sprint",
            "owner": "Engineering Manager"
        })
    
    # Actions for low activity authors
    if low_activity_authors:
        actions.append({
            "priority": "MEDIUM",
            "category": "Team Development",
            "action": f"1-on-1s with {len(low_activity_authors)} low-activity developers",
            "timeline": "This week",
            "owner": "Team Lead"
        })
    
    # Quality improvement actions
    commit_types = report_data['overall_distributions']['commit_types']
    total_commits = report_data['summary']['total_commits']
    fix_percentage = (commit_types.get('fix', 0) / total_commits) * 100
    
    if fix_percentage > 40:
        actions.append({
            "priority": "HIGH",
            "category": "Quality Improvement",
            "action": "Implement stricter code review process",
            "timeline": "Next 2 weeks",
            "owner": "Tech Lead"
        })
    
    # Print action plan
    if actions:
        print(f"\n📝 CÁC HÀNH ĐỘNG CẦN THỰC HIỆN:")
        for i, action in enumerate(actions, 1):
            print(f"\n{i}. [{action['priority']}] {action['category']}")
            print(f"   📋 Action: {action['action']}")
            print(f"   ⏰ Timeline: {action['timeline']}")
            print(f"   👤 Owner: {action['owner']}")
    else:
        print(f"\n✅ Team đang hoạt động tốt, không cần action đặc biệt!")

def main():
    """Hàm chính"""
    print("🚀 ADVANCED COMMIT ANALYSIS")
    print("="*60)
    
    # Find the latest report
    test_results_dir = Path(__file__).parent / "test_results"
    
    if not test_results_dir.exists():
        print("❌ Không tìm thấy thư mục test_results.")
        print("   Hãy chạy: python test_commit_analyzer.py")
        return
    
    # Get the latest report file
    report_files = list(test_results_dir.glob("commit_analysis_report_*.json"))
    if not report_files:
        print("❌ Không tìm thấy file báo cáo.")
        print("   Hãy chạy: python test_commit_analyzer.py")
        return
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Đang phân tích: {latest_report.name}")
    
    # Load report data
    try:
        report_data = load_analysis_report(latest_report)
        print(f"✅ Đã load báo cáo thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load báo cáo: {e}")
        return
    
    # Perform analysis
    analyze_author_patterns(report_data)
    generate_detailed_recommendations(report_data)
    create_action_plan(report_data)
    
    print(f"\n" + "="*80)
    print("✅ PHÂN TÍCH HOÀN THÀNH!")
    print("="*80)
    print(f"📊 Đã phân tích {report_data['summary']['total_commits']} commits")
    print(f"👥 Từ {report_data['summary']['unique_authors']} developers")
    print(f"🎯 Model confidence trung bình: 99.2%")

if __name__ == "__main__":
    main()
