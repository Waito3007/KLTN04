#!/usr/bin/env python3
"""
Advanced Commit Analyzer - Phân tích chi tiết và đưa ra khuyến nghị
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

def load_analysis_report(report_path):
    """Load báo cáo phân tích từ file JSON"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_visualizations(report_data, output_dir):
    """Tạo các biểu đồ phân tích"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Commit Type Distribution
    commit_types = report_data['overall_distributions']['commit_types']
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.pie(commit_types.values(), labels=commit_types.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Phân bố loại commit')
    
    # 2. Author Activity Levels
    activity_levels = report_data['activity_analysis']['activity_levels']
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(activity_levels.keys(), activity_levels.values())
    plt.title('Mức độ hoạt động của tác giả')
    plt.ylabel('Số lượng tác giả')
    
    # Color bars differently
    colors = ['red', 'orange', 'green', 'blue']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 3. Purpose Distribution
    purposes = report_data['overall_distributions']['purposes']
    
    plt.subplot(2, 2, 3)
    plt.barh(list(purposes.keys()), list(purposes.values()))
    plt.title('Phân bố mục đích commit')
    plt.xlabel('Số lượng')
    
    # 4. Sentiment Distribution
    sentiments = report_data['overall_distributions']['sentiments']
    
    plt.subplot(2, 2, 4)
    colors_sentiment = {'positive': 'green', 'neutral': 'gray', 'negative': 'red', 'urgent': 'orange'}
    sentiment_colors = [colors_sentiment.get(s, 'blue') for s in sentiments.keys()]
    plt.bar(sentiments.keys(), sentiments.values(), color=sentiment_colors)
    plt.title('Phân bố cảm xúc commit')
    plt.ylabel('Số lượng')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'commit_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Biểu đồ tổng quan đã được lưu vào: {output_dir / 'commit_analysis_overview.png'}")

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
            most_common_type = max(stats['commit_types'], key=stats['commit_types'].get)
            type_percentage = (stats['commit_types'][most_common_type] / stats['total_commits']) * 100
            print(f"   🏷️  Loại commit chủ yếu: {most_common_type} ({type_percentage:.1f}%)")
        
        # Phân tích purposes
        if stats['purposes']:
            most_common_purpose = max(stats['purposes'], key=stats['purposes'].get)
            purpose_percentage = (stats['purposes'][most_common_purpose] / stats['total_commits']) * 100
            print(f"   🎯 Mục đích chủ yếu: {most_common_purpose} ({purpose_percentage:.1f}%)")
        
        # Phân tích sentiment
        if stats['sentiments']:
            most_common_sentiment = max(stats['sentiments'], key=stats['sentiments'].get)
            sentiment_percentage = (stats['sentiments'][most_common_sentiment] / stats['total_commits']) * 100
            print(f"   😊 Cảm xúc chủ yếu: {most_common_sentiment} ({sentiment_percentage:.1f}%)")

def generate_recommendations(report_data):
    """Tạo khuyến nghị cho team"""
    print("\n" + "="*80)
    print("💡 KHUYẾN NGHỊ CHO TEAM")
    print("="*80)
    
    author_stats = report_data['author_statistics']
    overloaded_authors = report_data['activity_analysis']['overloaded_authors']
    low_activity_authors = report_data['activity_analysis']['low_activity_authors']
    
    # Khuyến nghị cho overloaded authors
    if overloaded_authors:
        print(f"\n🔥 TÌNH TRẠNG QUÁ TẢI ({len(overloaded_authors)} tác giả):")
        for author in overloaded_authors:
            stats = author_stats[author]
            print(f"   ⚠️  {author}: {stats['total_commits']} commits")
            print(f"      💡 Khuyến nghị: Cân nhắc phân phối công việc hoặc hỗ trợ thêm nhân lực")
            
            # Phân tích loại commit để đưa ra khuyến nghị cụ thể
            if stats['commit_types']:
                fix_count = stats['commit_types'].get('fix', 0)
                if fix_count > stats['total_commits'] * 0.4:
                    print(f"      🐛 Nhiều fix commits ({fix_count}): Cần review code kỹ hơn hoặc tăng cường testing")
                
                feat_count = stats['commit_types'].get('feat', 0)
                if feat_count > stats['total_commits'] * 0.6:
                    print(f"      ✨ Nhiều feature commits ({feat_count}): Tác giả có thể là key developer")
    
    # Khuyến nghị cho low activity authors
    if low_activity_authors:
        print(f"\n💤 HOẠT ĐỘNG THẤP ({len(low_activity_authors)} tác giả):")
        for author in low_activity_authors:
            stats = author_stats[author]
            print(f"   📉 {author}: {stats['total_commits']} commits")
            print(f"      💡 Khuyến nghị: Kiểm tra workload, cung cấp hỗ trợ hoặc training thêm")
    
    # Phân tích overall patterns
    print(f"\n📈 PHÂN TÍCH TỔNG QUAN:")
    
    commit_types = report_data['overall_distributions']['commit_types']
    total_commits = sum(commit_types.values())
    
    fix_percentage = (commit_types.get('fix', 0) / total_commits) * 100
    feat_percentage = (commit_types.get('feat', 0) / total_commits) * 100
    
    if fix_percentage > 40:
        print(f"   🐛 Tỷ lệ fix commits cao ({fix_percentage:.1f}%)")
        print(f"      💡 Khuyến nghị: Tăng cường code review, testing, và quality assurance")
    
    if feat_percentage < 30:
        print(f"   📦 Tỷ lệ feature commits thấp ({feat_percentage:.1f}%)")
        print(f"      💡 Khuyến nghị: Cân nhắc tăng tốc độ phát triển tính năng mới")
    
    # Sentiment analysis
    sentiments = report_data['overall_distributions']['sentiments']
    total_sentiments = sum(sentiments.values())
    
    negative_percentage = (sentiments.get('negative', 0) / total_sentiments) * 100
    urgent_percentage = (sentiments.get('urgent', 0) / total_sentiments) * 100
    
    if negative_percentage > 30:
        print(f"   😞 Tỷ lệ sentiment tiêu cực cao ({negative_percentage:.1f}%)")
        print(f"      💡 Khuyến nghị: Kiểm tra morale của team, cải thiện quy trình làm việc")
    
    if urgent_percentage > 10:
        print(f"   🚨 Tỷ lệ urgent commits cao ({urgent_percentage:.1f}%)")
        print(f"      💡 Khuyến nghị: Cải thiện planning và risk management")

def create_team_dashboard(report_data, output_dir):
    """Tạo dashboard tổng quan cho team"""
    output_dir = Path(output_dir)
    
    # Create a comprehensive team report
    dashboard_data = {
        "timestamp": datetime.now().isoformat(),
        "team_health": {
            "total_authors": len(report_data['author_statistics']),
            "total_commits": report_data['summary']['total_commits'],
            "avg_commits_per_author": report_data['summary']['avg_commits_per_author']
        },
        "risk_indicators": {
            "overloaded_authors": len(report_data['activity_analysis']['overloaded_authors']),
            "low_activity_authors": len(report_data['activity_analysis']['low_activity_authors']),
            "fix_percentage": (report_data['overall_distributions']['commit_types'].get('fix', 0) / 
                             report_data['summary']['total_commits']) * 100
        },
        "recommendations": []
    }
    
    # Add recommendations
    if dashboard_data['risk_indicators']['overloaded_authors'] > 0:
        dashboard_data['recommendations'].append({
            "type": "workload_balancing",
            "priority": "high",
            "message": f"Có {dashboard_data['risk_indicators']['overloaded_authors']} tác giả bị quá tải"
        })
    
    if dashboard_data['risk_indicators']['fix_percentage'] > 40:
        dashboard_data['recommendations'].append({
            "type": "quality_improvement",
            "priority": "medium",
            "message": f"Tỷ lệ fix commits cao ({dashboard_data['risk_indicators']['fix_percentage']:.1f}%)"
        })
    
    # Save dashboard
    dashboard_file = output_dir / 'team_dashboard.json'
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Team dashboard đã được lưu vào: {dashboard_file}")

def main():
    """Hàm chính để phân tích nâng cao"""
    print("🚀 ADVANCED COMMIT ANALYSIS")
    print("="*60)
    
    # Find the latest report
    test_results_dir = Path(__file__).parent / "test_results"
    
    if not test_results_dir.exists():
        print("❌ Không tìm thấy thư mục test_results. Hãy chạy test_commit_analyzer.py trước.")
        return
    
    # Get the latest report file
    report_files = list(test_results_dir.glob("commit_analysis_report_*.json"))
    if not report_files:
        print("❌ Không tìm thấy file báo cáo. Hãy chạy test_commit_analyzer.py trước.")
        return
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Đang phân tích: {latest_report.name}")
    
    # Load report data
    report_data = load_analysis_report(latest_report)
    
    # Create output directory for advanced analysis
    advanced_output_dir = test_results_dir / "advanced_analysis"
    advanced_output_dir.mkdir(exist_ok=True)
    
    # Perform advanced analysis
    analyze_author_patterns(report_data)
    generate_recommendations(report_data)
    
    # Create visualizations
    try:
        create_visualizations(report_data, advanced_output_dir)
    except Exception as e:
        print(f"⚠️  Không thể tạo biểu đồ: {e}")
    
    # Create team dashboard
    create_team_dashboard(report_data, advanced_output_dir)
    
    print(f"\n✅ Phân tích nâng cao hoàn thành!")
    print(f"📁 Kết quả lưu tại: {advanced_output_dir}")

if __name__ == "__main__":
    main()
