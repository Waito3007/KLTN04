#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST MÔ HÌNH HAN - MINH HỌA KHẢ NĂNG PHÂN LOẠI
(Không load model thực, chỉ demo flow hoạt động)
"""

import os
from datetime import datetime

def simulate_han_model_prediction(commit_message):
    """
    Mô phỏng kết quả từ model HAN thực
    (Thực tế sẽ load từ best_model.pth)
    """
    text = commit_message.lower()
    
    # Mô phỏng logic phân loại của HAN model
    
    # 1. Commit Type Classification
    if any(word in text for word in ['feat:', 'feature:', 'add', 'implement', 'create']):
        commit_type = 'feat'
        type_confidence = 0.95
    elif any(word in text for word in ['fix:', 'bug:', 'resolve', 'patch']):
        commit_type = 'fix'  
        type_confidence = 0.92
    elif any(word in text for word in ['docs:', 'documentation', 'readme']):
        commit_type = 'docs'
        type_confidence = 0.89
    elif any(word in text for word in ['test:', 'testing', 'spec']):
        commit_type = 'test'
        type_confidence = 0.87
    elif any(word in text for word in ['refactor:', 'restructure', 'cleanup']):
        commit_type = 'refactor'
        type_confidence = 0.91
    elif any(word in text for word in ['chore:', 'update', 'dependency']):
        commit_type = 'chore'
        type_confidence = 0.88
    elif any(word in text for word in ['style:', 'format', 'lint']):
        commit_type = 'style'
        type_confidence = 0.86
    elif any(word in text for word in ['perf:', 'performance', 'optimize']):
        commit_type = 'perf'
        type_confidence = 0.93
    else:
        commit_type = 'other'
        type_confidence = 0.75
    
    # 2. Purpose Classification
    purpose_map = {
        'feat': 'Feature Implementation',
        'fix': 'Bug Fix',
        'docs': 'Documentation Update', 
        'test': 'Test Update',
        'refactor': 'Code Refactoring',
        'chore': 'Maintenance',
        'style': 'Code Style',
        'perf': 'Performance Improvement',
        'other': 'Other'
    }
    purpose = purpose_map.get(commit_type, 'Other')
    purpose_confidence = type_confidence - 0.03
    
    # 3. Sentiment Analysis
    if any(word in text for word in ['critical', 'urgent', 'emergency', 'severe']):
        sentiment = 'urgent'
        sentiment_confidence = 0.94
    elif any(word in text for word in ['error', 'bug', 'fail', 'problem']):
        sentiment = 'negative'
        sentiment_confidence = 0.88
    elif any(word in text for word in ['improve', 'enhance', 'optimize', 'add', 'new']):
        sentiment = 'positive'
        sentiment_confidence = 0.90
    else:
        sentiment = 'neutral'
        sentiment_confidence = 0.85
    
    # 4. Tech Tag Classification (mở rộng)
    if any(word in text for word in ['auth', 'authentication', 'login', 'oauth']):
        tech_tag = 'authentication'
        tech_confidence = 0.92
    elif any(word in text for word in ['database', 'db', 'sql', 'query']):
        tech_tag = 'database'
        tech_confidence = 0.89
    elif any(word in text for word in ['api', 'endpoint', 'rest']):
        tech_tag = 'api'
        tech_confidence = 0.91
    elif any(word in text for word in ['ui', 'frontend', 'component']):
        tech_tag = 'frontend'
        tech_confidence = 0.87
    elif any(word in text for word in ['security', 'vulnerability', 'encryption']):
        tech_tag = 'security'
        tech_confidence = 0.95
    else:
        tech_tag = 'general'
        tech_confidence = 0.80
    
    return {
        'commit_type': {'label': commit_type, 'confidence': type_confidence},
        'purpose': {'label': purpose, 'confidence': purpose_confidence},
        'sentiment': {'label': sentiment, 'confidence': sentiment_confidence},
        'tech_tag': {'label': tech_tag, 'confidence': tech_confidence}
    }

def run_han_model_demo():
    """Demo khả năng phân loại của model HAN với phân tích chi tiết"""
    
    print("=" * 80)
    print("🤖 DEMO MÔ HÌNH HAN - PHÂN TÍCH COMMIT CHI TIẾT")
    print("=" * 80)
    print(f"⏰ Thời gian demo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📝 LƯU Ý: Demo này mô phỏng kết quả từ model HAN thực")
    print("🔧 Model thực được lưu tại: models/han_github_model/best_model.pth")
    print()
    
    # Test cases đa dạng với 30 commits và tác giả
    test_commits = [
        # Developer 1: John Smith (Frontend specialist)
        ("john.smith@company.com", "feat: implement responsive navigation component"),
        ("john.smith@company.com", "feat: add dark mode toggle functionality"),
        ("john.smith@company.com", "fix: resolve mobile layout issues in header"),
        ("john.smith@company.com", "style: update CSS variables for consistent theming"),
        ("john.smith@company.com", "feat: create user profile management page"),
        
        # Developer 2: Sarah Johnson (Backend specialist)  
        ("sarah.johnson@company.com", "feat: implement user authentication API"),
        ("sarah.johnson@company.com", "fix: resolve database connection timeout issues"),
        ("sarah.johnson@company.com", "feat: add JWT token refresh mechanism"),
        ("sarah.johnson@company.com", "perf: optimize database queries for user search"),
        ("sarah.johnson@company.com", "fix: handle edge case in password validation"),
        ("sarah.johnson@company.com", "feat: implement role-based access control"),
        
        # Developer 3: Mike Chen (DevOps/Infrastructure)
        ("mike.chen@company.com", "chore: update Docker configuration for production"),
        ("mike.chen@company.com", "fix: resolve CI/CD pipeline deployment issues"),
        ("mike.chen@company.com", "chore: upgrade Node.js to version 18 LTS"),
        ("mike.chen@company.com", "perf: optimize build process with caching"),
        
        # Developer 4: Emily Davis (QA/Testing)
        ("emily.davis@company.com", "test: add unit tests for authentication service"),
        ("emily.davis@company.com", "test: implement integration tests for API endpoints"),
        ("emily.davis@company.com", "fix: correct test data setup for user scenarios"),
        ("emily.davis@company.com", "test: add end-to-end tests for login flow"),
        
        # Developer 5: Alex Rodriguez (Security specialist)
        ("alex.rodriguez@company.com", "fix(security): patch XSS vulnerability in user input"),
        ("alex.rodriguez@company.com", "feat(security): implement rate limiting for API"),
        ("alex.rodriguez@company.com", "fix(security): resolve CSRF token validation issue"),
        
        # Developer 6: Lisa Wang (Documentation)
        ("lisa.wang@company.com", "docs: update API documentation with new endpoints"),
        ("lisa.wang@company.com", "docs: add installation guide for development setup"),
        ("lisa.wang@company.com", "docs: create user manual for admin features"),
        
        # Developer 7: Tom Brown (Performance specialist)
        ("tom.brown@company.com", "perf: implement lazy loading for large datasets"),
        ("tom.brown@company.com", "perf: optimize image compression and caching"),
        ("tom.brown@company.com", "refactor: simplify complex rendering logic"),
        
        # Developer 8: Anna Kim (Junior developer - fewer commits)
        ("anna.kim@company.com", "fix: correct typo in error messages"),
        ("anna.kim@company.com", "style: fix indentation in configuration files")
    ]    
    print("🧪 BẮT ĐẦU DEMO VỚI 30 COMMIT MESSAGES")
    print("=" * 80)
    
    total_tests = len(test_commits)
    author_stats = {}
    commit_type_stats = {}
    purpose_stats = {}
    sentiment_stats = {}
    tech_tag_stats = {}
    
    for i, (author, commit_message) in enumerate(test_commits, 1):
        print(f"\n🔍 DEMO #{i}")
        print("-" * 60)
        
        # Input
        print(f"📝 ĐẦU VÀO:")
        print(f"   Author: {author}")
        print(f"   Commit Message: '{commit_message}'")
        
        # Model prediction (simulated)
        predictions = simulate_han_model_prediction(commit_message)
        
        print(f"\n🤖 KẾT QUẢ TỪ MODEL HAN:")
        print(f"   📋 Commit Type: {predictions['commit_type']['label']} "
              f"(tin cậy: {predictions['commit_type']['confidence']:.0%})")
        print(f"   🎯 Purpose: {predictions['purpose']['label']} "
              f"(tin cậy: {predictions['purpose']['confidence']:.0%})")
        print(f"   😊 Sentiment: {predictions['sentiment']['label']} "
              f"(tin cậy: {predictions['sentiment']['confidence']:.0%})")
        print(f"   🏷️ Tech Tag: {predictions['tech_tag']['label']} "
              f"(tin cậy: {predictions['tech_tag']['confidence']:.0%})")
        
        # Phân tích
        expected_type = commit_message.split(':')[0].split('(')[0]
        predicted_type = predictions['commit_type']['label']
        is_correct = expected_type.lower() == predicted_type.lower()
        
        print(f"\n✅ PHÂN TÍCH:")
        print(f"   Expected: {expected_type}")
        print(f"   Predicted: {predicted_type}")
        print(f"   Kết quả: {'✓ CHÍNH XÁC' if is_correct else '✗ SAI SÓT'}")
        
        # Thu thập thống kê
        if author not in author_stats:
            author_stats[author] = {
                'total_commits': 0,
                'commit_types': {},
                'purposes': {},
                'sentiments': {},
                'tech_tags': {}
            }
        
        author_stats[author]['total_commits'] += 1
        
        # Thống kê theo loại commit
        commit_type = predictions['commit_type']['label']
        author_stats[author]['commit_types'][commit_type] = author_stats[author]['commit_types'].get(commit_type, 0) + 1
        commit_type_stats[commit_type] = commit_type_stats.get(commit_type, 0) + 1
        
        # Thống kê theo purpose
        purpose = predictions['purpose']['label']
        author_stats[author]['purposes'][purpose] = author_stats[author]['purposes'].get(purpose, 0) + 1
        purpose_stats[purpose] = purpose_stats.get(purpose, 0) + 1
        
        # Thống kê theo sentiment
        sentiment = predictions['sentiment']['label']
        author_stats[author]['sentiments'][sentiment] = author_stats[author]['sentiments'].get(sentiment, 0) + 1
        sentiment_stats[sentiment] = sentiment_stats.get(sentiment, 0) + 1
        
        # Thống kê theo tech tag
        tech_tag = predictions['tech_tag']['label']
        author_stats[author]['tech_tags'][tech_tag] = author_stats[author]['tech_tags'].get(tech_tag, 0) + 1
        tech_tag_stats[tech_tag] = tech_tag_stats.get(tech_tag, 0) + 1
        
        print("-" * 60)
    
    # Tổng kết và phân tích chi tiết
    print(f"\n📊 TỔNG KẾT DEMO & PHÂN TÍCH CHI TIẾT")
    print("=" * 80)
    print(f"🔢 Tổng số commits demo: {total_tests}")
    print(f"👥 Tổng số developers: {len(author_stats)}")
    print()
    
    # Phân tích theo tác giả
    print("👤 PHÂN TÍCH THEO TÁC GIẢ:")
    print("=" * 60)
    
    # Sắp xếp theo số commit (từ nhiều đến ít)
    sorted_authors = sorted(author_stats.items(), key=lambda x: x[1]['total_commits'], reverse=True)
    
    for author, stats in sorted_authors:
        name = author.split('@')[0].replace('.', ' ').title()
        print(f"\n🧑‍💻 {name} ({author})")
        print(f"   📊 Tổng commits: {stats['total_commits']}")
        
        # Top commit types
        top_commit_types = sorted(stats['commit_types'].items(), key=lambda x: x[1], reverse=True)
        print(f"   🏷️ Commit types:")
        for commit_type, count in top_commit_types:
            percentage = (count / stats['total_commits']) * 100
            print(f"      • {commit_type}: {count} lần ({percentage:.1f}%)")
        
        # Top purposes
        top_purposes = sorted(stats['purposes'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   🎯 Top purposes:")
        for purpose, count in top_purposes:
            print(f"      • {purpose}: {count} lần")
        
        # Dominant tech tags
        top_tech_tags = sorted(stats['tech_tags'].items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"   🔧 Tech focus:")
        for tech_tag, count in top_tech_tags:
            print(f"      • {tech_tag}: {count} lần")
    
    print("\n" + "=" * 60)
    
    # Phân tích tổng quan
    print("\n📈 THỐNG KÊ TỔNG QUAN:")
    print("=" * 60)
    
    # Top commit types
    print("\n🏷️ PHÂN BỐ COMMIT TYPES:")
    sorted_commit_types = sorted(commit_type_stats.items(), key=lambda x: x[1], reverse=True)
    for commit_type, count in sorted_commit_types:
        percentage = (count / total_tests) * 100
        print(f"   • {commit_type}: {count} commits ({percentage:.1f}%)")
    
    # Top purposes
    print("\n🎯 PHÂN BỐ PURPOSES:")
    sorted_purposes = sorted(purpose_stats.items(), key=lambda x: x[1], reverse=True)
    for purpose, count in sorted_purposes[:5]:  # Top 5
        percentage = (count / total_tests) * 100
        print(f"   • {purpose}: {count} commits ({percentage:.1f}%)")
    
    # Sentiment analysis
    print("\n😊 PHÂN BỐ SENTIMENT:")
    sorted_sentiments = sorted(sentiment_stats.items(), key=lambda x: x[1], reverse=True)
    for sentiment, count in sorted_sentiments:
        percentage = (count / total_tests) * 100
        print(f"   • {sentiment}: {count} commits ({percentage:.1f}%)")
    
    # Tech tags
    print("\n🔧 PHÂN BỐ TECH TAGS:")
    sorted_tech_tags = sorted(tech_tag_stats.items(), key=lambda x: x[1], reverse=True)
    for tech_tag, count in sorted_tech_tags:
        percentage = (count / total_tests) * 100
        print(f"   • {tech_tag}: {count} commits ({percentage:.1f}%)")
    
    # Insights và recommendations
    print("\n💡 INSIGHTS & NHẬN XÉT:")
    print("=" * 60)
    
    # Developer với nhiều commits nhất
    most_active = sorted_authors[0]
    least_active = sorted_authors[-1]
    
    print(f"🏆 Developer hoạt động nhất: {most_active[0].split('@')[0].replace('.', ' ').title()}")
    print(f"   • {most_active[1]['total_commits']} commits ({(most_active[1]['total_commits']/total_tests)*100:.1f}% tổng commits)")
    
    print(f"\n📉 Developer ít commits nhất: {least_active[0].split('@')[0].replace('.', ' ').title()}")
    print(f"   • {least_active[1]['total_commits']} commits ({(least_active[1]['total_commits']/total_tests)*100:.1f}% tổng commits)")
    
    # Phân tích xu hướng
    feat_count = commit_type_stats.get('feat', 0)
    fix_count = commit_type_stats.get('fix', 0)
    
    print(f"\n🔍 Phân tích xu hướng:")
    print(f"   • Tỷ lệ feat/fix: {feat_count}:{fix_count}")
    if feat_count > fix_count:
        print("   • Team đang focus vào phát triển tính năng mới")
    elif fix_count > feat_count:
        print("   • Team đang focus vào sửa lỗi và ổn định hệ thống")
    else:
        print("   • Team có sự cân bằng giữa phát triển và maintenance")
    
    print(f"\n📈 Model HAN có thể phân loại: 4 tasks đồng thời")
    print(f"🎯 Các tasks:")
    print(f"   • Commit Type (feat, fix, docs, test, refactor, etc.)")
    print(f"   • Purpose (Feature Implementation, Bug Fix, etc.)")
    print(f"   • Sentiment (positive, negative, neutral, urgent)")
    print(f"   • Tech Tag (authentication, database, api, etc.)")
    print()
    print(f"⚡ Ưu điểm của Model HAN:")
    print(f"   ✓ Multi-task learning (4 tasks cùng lúc)")
    print(f"   ✓ Hierarchical attention (word-level + sentence-level)")
    print(f"   ✓ High accuracy trên training data (~99%)")
    print(f"   ✓ Hỗ trợ conventional commit format")
    print(f"   ✓ Phân tích được patterns của từng developer")
    print()
    print(f"🔧 Sử dụng Model thực:")
    print(f"   1. Load từ: models/han_github_model/best_model.pth")
    print(f"   2. Thay thế simulate_han_model_prediction() bằng model thực")
    print(f"   3. Sử dụng tokenizer và label_encoders từ checkpoint")
    print(f"\n🎉 DEMO HOÀN THÀNH!")
    print("=" * 80)
    
    # Tạo và lưu báo cáo chi tiết
    print(f"\n📄 TẠO BÁO CÁO CHI TIẾT...")
    detailed_report = generate_detailed_report(
        author_stats, commit_type_stats, purpose_stats, 
        sentiment_stats, tech_tag_stats, total_tests
    )
    
    # Lưu báo cáo
    report_saved = save_analysis_report(detailed_report)
    
    if report_saved:
        print(f"✅ Báo cáo phân tích đã được tạo thành công!")
        print(f"📊 Có thể sử dụng báo cáo này để:")
        print(f"   • Đánh giá hiệu suất team")
        print(f"   • Phân tích xu hướng phát triển")
        print(f"   • Lập kế hoạch phân công công việc")
        print(f"   • Training và mentoring developers")
    
    return detailed_report

def generate_detailed_report(author_stats, commit_type_stats, purpose_stats, sentiment_stats, tech_tag_stats, total_commits):
    """Tạo báo cáo chi tiết về phân tích commits"""
    
    report = {
        'summary': {
            'total_commits': total_commits,
            'total_developers': len(author_stats),
            'analysis_date': datetime.now().isoformat()
        },
        'developer_analysis': {},
        'overall_statistics': {
            'commit_types': commit_type_stats,
            'purposes': purpose_stats,
            'sentiments': sentiment_stats,
            'tech_tags': tech_tag_stats
        },
        'insights': {}
    }
    
    # Phân tích chi tiết từng developer
    sorted_authors = sorted(author_stats.items(), key=lambda x: x[1]['total_commits'], reverse=True)
    
    for author, stats in sorted_authors:
        name = author.split('@')[0].replace('.', ' ').title()
        
        # Tìm commit type chủ đạo
        main_commit_type = max(stats['commit_types'].items(), key=lambda x: x[1])
        
        # Tính productivity score (commits per category diversity)
        diversity_score = len(stats['commit_types']) / len(commit_type_stats) * 100
        
        report['developer_analysis'][author] = {
            'name': name,
            'total_commits': stats['total_commits'],
            'commit_percentage': (stats['total_commits'] / total_commits) * 100,
            'main_commit_type': main_commit_type[0],
            'main_commit_type_count': main_commit_type[1],
            'diversity_score': diversity_score,
            'specialization': 'Specialist' if diversity_score < 40 else 'Generalist',
            'detailed_stats': stats
        }
    
    # Insights tổng quan
    most_active = sorted_authors[0]
    least_active = sorted_authors[-1]
    feat_count = commit_type_stats.get('feat', 0)
    fix_count = commit_type_stats.get('fix', 0)
    
    report['insights'] = {
        'most_active_developer': {
            'email': most_active[0],
            'name': most_active[0].split('@')[0].replace('.', ' ').title(),
            'commits': most_active[1]['total_commits']
        },
        'least_active_developer': {
            'email': least_active[0],
            'name': least_active[0].split('@')[0].replace('.', ' ').title(),
            'commits': least_active[1]['total_commits']
        },
        'team_focus': 'Feature Development' if feat_count > fix_count else 'Bug Fixing' if fix_count > feat_count else 'Balanced',
        'feat_fix_ratio': f"{feat_count}:{fix_count}",
        'productivity_distribution': 'Balanced' if max(author_stats.values(), key=lambda x: x['total_commits'])['total_commits'] <= total_commits * 0.4 else 'Concentrated'
    }
    
    return report

def save_analysis_report(report, filename="commit_analysis_detailed_report.json"):
    """Lưu báo cáo phân tích ra file JSON"""
    import json
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📄 Báo cáo đã được lưu: {filename}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu báo cáo: {e}")
        return False

def main():
    """Hàm chính"""
    try:
        detailed_report = run_han_model_demo()
        
        # Hiển thị một số insights quan trọng
        print(f"\n🎯 INSIGHTS QUAN TRỌNG:")
        print(f"=" * 50)
        
        insights = detailed_report['insights']
        print(f"👑 Developer tích cực nhất: {insights['most_active_developer']['name']}")
        print(f"   ({insights['most_active_developer']['commits']} commits)")
        
        print(f"📉 Developer ít commit nhất: {insights['least_active_developer']['name']}")
        print(f"   ({insights['least_active_developer']['commits']} commits)")
        
        print(f"🎯 Focus của team: {insights['team_focus']}")
        print(f"⚖️ Tỷ lệ feat/fix: {insights['feat_fix_ratio']}")
        print(f"📊 Phân bố productivity: {insights['productivity_distribution']}")
        
        print(f"\n💼 GỢI Ý QUẢN LÝ TEAM:")
        print(f"=" * 50)
        
        # Phân tích và đưa ra gợi ý
        dev_analysis = detailed_report['developer_analysis']
        specialists = [dev for dev in dev_analysis.values() if dev['specialization'] == 'Specialist']
        generalists = [dev for dev in dev_analysis.values() if dev['specialization'] == 'Generalist']
        
        print(f"🔧 Specialists ({len(specialists)} người): Focus sâu vào 1-2 lĩnh vực")
        for dev in specialists[:3]:  # Top 3
            print(f"   • {dev['name']}: chuyên {dev['main_commit_type']} ({dev['main_commit_type_count']} commits)")
        
        print(f"🌐 Generalists ({len(generalists)} người): Đa dạng nhiều lĩnh vực")
        for dev in generalists[:3]:  # Top 3
            print(f"   • {dev['name']}: diversity score {dev['diversity_score']:.1f}%")
        
        return detailed_report
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy demo: {e}")
        return None

if __name__ == "__main__":
    main()
