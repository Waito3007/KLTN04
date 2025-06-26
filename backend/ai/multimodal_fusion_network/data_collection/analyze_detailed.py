"""
Script phân tích nâng cao dữ liệu commit đã thu thập.
Chạy script này để đánh giá chất lượng, thống kê chi tiết và tạo báo cáo trực quan.
"""
import os
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_commit_data(file_path: str) -> Tuple[List[Dict], Dict]:
    """
    Đọc dữ liệu commit từ file JSON.
    
    Args:
        file_path: Đường dẫn đến file dữ liệu
        
    Returns:
        Tuple chứa danh sách commit và metadata
    """
    logger.info(f"Đọc dữ liệu từ {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        commits = data.get('data', [])
        metadata = data.get('metadata', {})
        
        logger.info(f"Đã đọc {len(commits)} commit")
        return commits, metadata
    
    except Exception as e:
        logger.error(f"Lỗi khi đọc file: {str(e)}")
        return [], {}

def analyze_commit_length(commits: List[Dict]) -> Dict:
    """
    Phân tích độ dài của commit message.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về độ dài commit message
    """
    lengths = [len(commit.get('text', '').split()) for commit in commits]
    
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'std': np.std(lengths),
        'distribution': {
            'very_short (1-5)': sum(1 for l in lengths if 1 <= l <= 5),
            'short (6-15)': sum(1 for l in lengths if 6 <= l <= 15),
            'medium (16-30)': sum(1 for l in lengths if 16 <= l <= 30),
            'long (31-50)': sum(1 for l in lengths if 31 <= l <= 50),
            'very_long (>50)': sum(1 for l in lengths if l > 50)
        }
    }

def analyze_commits_by_time(commits: List[Dict]) -> Dict:
    """
    Phân tích commit theo thời gian.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê commit theo thời gian
    """
    commits_by_year = Counter()
    commits_by_month = Counter()
    commits_by_hour = Counter()
    commits_by_day = Counter()
    
    for commit in commits:
        timestamp = commit.get('metadata', {}).get('timestamp', '')
        if not timestamp:
            continue
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            commits_by_year[dt.year] += 1
            commits_by_month[dt.strftime('%Y-%m')] += 1
            commits_by_hour[dt.hour] += 1
            commits_by_day[dt.strftime('%A')] += 1
        except (ValueError, TypeError):
            pass
    
    return {
        'by_year': dict(commits_by_year.most_common()),
        'by_month': dict(commits_by_month.most_common()[-12:]),  # 12 tháng gần nhất
        'by_hour': dict(sorted(commits_by_hour.items())),
        'by_day': dict(commits_by_day.most_common())
    }

def analyze_commit_changes(commits: List[Dict]) -> Dict:
    """
    Phân tích thay đổi trong commit.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về các thay đổi trong commit
    """
    changes = [commit.get('metadata', {}).get('total_changes', 0) for commit in commits]
    files_changed = [commit.get('metadata', {}).get('files_changed', 0) for commit in commits]
    additions = [commit.get('metadata', {}).get('additions', 0) for commit in commits]
    deletions = [commit.get('metadata', {}).get('deletions', 0) for commit in commits]
    
    changes_stats = {
        'mean_changes': np.mean(changes),
        'median_changes': np.median(changes),
        'max_changes': max(changes),
        'mean_files': np.mean(files_changed),
        'median_files': np.median(files_changed),
        'max_files': max(files_changed),
        'mean_additions': np.mean(additions),
        'median_additions': np.median(additions),
        'mean_deletions': np.mean(deletions),
        'median_deletions': np.median(deletions),
        'changes_distribution': {
            'small (1-10)': sum(1 for c in changes if 1 <= c <= 10),
            'medium (11-50)': sum(1 for c in changes if 11 <= c <= 50),
            'large (51-200)': sum(1 for c in changes if 51 <= c <= 200),
            'very_large (>200)': sum(1 for c in changes if c > 200)
        },
        'files_distribution': {
            '1 file': sum(1 for f in files_changed if f == 1),
            '2-3 files': sum(1 for f in files_changed if 2 <= f <= 3),
            '4-10 files': sum(1 for f in files_changed if 4 <= f <= 10),
            '>10 files': sum(1 for f in files_changed if f > 10)
        }
    }
    
    return changes_stats

def analyze_repositories(commits: List[Dict]) -> Dict:
    """
    Phân tích repository.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về repository
    """
    repos = Counter([commit.get('metadata', {}).get('repository', 'unknown') for commit in commits])
    commits_per_repo = {}
    authors_per_repo = defaultdict(set)
    
    for commit in commits:
        repo = commit.get('metadata', {}).get('repository', 'unknown')
        author = commit.get('metadata', {}).get('author', 'unknown')
        authors_per_repo[repo].add(author)
    
    # Chuyển set thành số lượng
    authors_count = {repo: len(authors) for repo, authors in authors_per_repo.items()}
    
    return {
        'total_repos': len(repos),
        'commits_per_repo': dict(repos.most_common(20)),  # Top 20
        'authors_per_repo': dict(sorted(authors_count.items(), key=lambda x: x[1], reverse=True)[:20])  # Top 20
    }

def analyze_file_types(commits: List[Dict]) -> Dict:
    """
    Phân tích loại file trong commit.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về loại file
    """
    all_file_types = Counter()
    
    for commit in commits:
        file_types = commit.get('metadata', {}).get('file_types', {})
        for ftype, count in file_types.items():
            all_file_types[ftype] += count
    
    # Chuyển đổi định dạng file thành ngôn ngữ lập trình
    language_mapping = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.jsx': 'React',
        '.tsx': 'React TypeScript',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C/C++ Header',
        '.hpp': 'C++ Header',
        '.cs': 'C#',
        '.go': 'Go',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.rs': 'Rust',
        '.scala': 'Scala',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'Sass',
        '.less': 'Less',
        '.md': 'Markdown',
        '.json': 'JSON',
        '.xml': 'XML',
        '.yml': 'YAML',
        '.yaml': 'YAML',
        '.sh': 'Shell',
        '.bat': 'Batch',
        '.ps1': 'PowerShell',
        '.sql': 'SQL',
        '.r': 'R',
        '.ipynb': 'Jupyter Notebook',
        '.dart': 'Dart',
        '.lua': 'Lua',
        '.clj': 'Clojure',
        '.ex': 'Elixir',
        '.exs': 'Elixir',
        '.erl': 'Erlang',
        '.hs': 'Haskell',
        '.pl': 'Perl',
        '.jl': 'Julia',
        '.fs': 'F#',
        '.vue': 'Vue',
        '.elm': 'Elm',
        '.coffee': 'CoffeeScript',
        '.tf': 'Terraform',
        '.gradle': 'Gradle',
        '.toml': 'TOML',
        '.ini': 'INI',
        '.config': 'Configuration',
        '.csproj': '.NET Project',
        '.vbproj': '.NET Project',
        '.fsproj': '.NET Project',
        '.sln': 'Solution File',
        '.gitignore': 'Git Config',
        '.dockerignore': 'Docker Config',
        '.dockerfile': 'Dockerfile',
        'no_extension': 'No Extension'
    }
    
    # Nhóm các loại file theo ngôn ngữ
    languages = Counter()
    for ftype, count in all_file_types.items():
        language = language_mapping.get(ftype.lower(), ftype)
        languages[language] += count
    
    return {
        'file_types': dict(all_file_types.most_common(30)),  # Top 30
        'languages': dict(languages.most_common(20))  # Top 20
    }

def analyze_merge_commits(commits: List[Dict]) -> Dict:
    """
    Phân tích commit merge.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về commit merge
    """
    merge_commits = [commit for commit in commits if commit.get('metadata', {}).get('is_merge', False)]
    merge_authors = Counter([commit.get('metadata', {}).get('author', 'unknown') for commit in merge_commits])
    merge_repos = Counter([commit.get('metadata', {}).get('repository', 'unknown') for commit in merge_commits])
    
    return {
        'total_merges': len(merge_commits),
        'percentage': len(merge_commits) / len(commits) * 100 if commits else 0,
        'top_merge_authors': dict(merge_authors.most_common(10)),
        'top_merge_repos': dict(merge_repos.most_common(10))
    }

def analyze_authors(commits: List[Dict]) -> Dict:
    """
    Phân tích tác giả commit.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về tác giả
    """
    authors = Counter([commit.get('metadata', {}).get('author', 'unknown') for commit in commits])
    emails = Counter([commit.get('metadata', {}).get('author_email', 'unknown') for commit in commits])
    
    # Phân tích số lượng commit theo tác giả
    author_commit_counts = {}
    for author, count in authors.items():
        if count >= 5:  # Chỉ xem xét tác giả có ít nhất 5 commit
            author_commit_counts[author] = count
    
    # Phân tích thời gian commit theo tác giả
    author_times = defaultdict(list)
    for commit in commits:
        author = commit.get('metadata', {}).get('author', 'unknown')
        timestamp = commit.get('metadata', {}).get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                author_times[author].append(dt.hour)
            except (ValueError, TypeError):
                pass
    
    # Tính giờ trung bình commit của mỗi tác giả
    author_avg_hours = {}
    for author, hours in author_times.items():
        if len(hours) >= 5:  # Chỉ xem xét tác giả có ít nhất 5 commit
            author_avg_hours[author] = np.mean(hours)
    
    return {
        'total_authors': len(authors),
        'top_authors': dict(authors.most_common(20)),
        'top_emails': dict(emails.most_common(20)),
        'author_distribution': {
            '1 commit': sum(1 for count in authors.values() if count == 1),
            '2-5 commits': sum(1 for count in authors.values() if 2 <= count <= 5),
            '6-20 commits': sum(1 for count in authors.values() if 6 <= count <= 20),
            '21-100 commits': sum(1 for count in authors.values() if 21 <= count <= 100),
            '>100 commits': sum(1 for count in authors.values() if count > 100)
        }
    }

def analyze_commit_quality(commits: List[Dict]) -> Dict:
    """
    Phân tích chất lượng commit message.
    
    Args:
        commits: Danh sách commit
    
    Returns:
        Thống kê về chất lượng commit
    """
    # Tìm kiếm một số mẫu thường gặp trong commit kém chất lượng
    low_quality_patterns = [
        'fix', 'fixes', 'fixed', 'fixing',
        'update', 'updates', 'updated', 'updating',
        'add', 'adds', 'added', 'adding',
        'remove', 'removes', 'removed', 'removing',
        'change', 'changes', 'changed', 'changing',
        'refactor', 'refactors', 'refactored', 'refactoring',
        'cleanup', 'clean up', 'cleaned up', 'cleaning up',
        'wip', 'WIP', 'work in progress',
        'test', 'tests', 'testing', 'tested',
        'merge', 'merged', 'merging'
    ]
    
    # Đếm số lượng commit chỉ chứa các mẫu đơn giản
    low_quality_count = 0
    single_word_count = 0
    
    for commit in commits:
        message = commit.get('text', '').lower().strip()
        
        # Kiểm tra commit một từ
        if len(message.split()) <= 1:
            single_word_count += 1
            low_quality_count += 1
            continue
        
        # Kiểm tra commit chỉ chứa các mẫu đơn giản
        is_low_quality = False
        for pattern in low_quality_patterns:
            if message == pattern or message.startswith(pattern + ':') or message.startswith(pattern + ' '):
                words = message.split()
                # Nếu chỉ có mẫu hoặc mẫu + 1-2 từ
                if len(words) <= 3:
                    is_low_quality = True
                    break
        
        if is_low_quality:
            low_quality_count += 1
    
    # Tính tỷ lệ commit chất lượng cao
    total_commits = len(commits)
    high_quality_count = total_commits - low_quality_count
    
    return {
        'total_analyzed': total_commits,
        'high_quality_count': high_quality_count,
        'low_quality_count': low_quality_count,
        'single_word_count': single_word_count,
        'high_quality_percentage': (high_quality_count / total_commits * 100) if total_commits else 0,
        'quality_distribution': {
            'high_quality': high_quality_count,
            'low_quality': low_quality_count,
            'single_word': single_word_count
        }
    }

def generate_visualizations(stats: Dict, output_dir: str) -> None:
    """
    Tạo các biểu đồ trực quan từ thống kê.
    
    Args:
        stats: Thống kê đã được phân tích
        output_dir: Thư mục đầu ra cho các biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Thiết lập kích thước hình ảnh mặc định
    plt.figure(figsize=(12, 6))
    
    # 1. Biểu đồ phân bố độ dài commit
    plt.figure(figsize=(10, 6))
    labels = list(stats['commit_length']['distribution'].keys())
    values = list(stats['commit_length']['distribution'].values())
    plt.bar(labels, values, color='skyblue')
    plt.title('Phân bố độ dài commit message')
    plt.ylabel('Số lượng commit')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_commit_length_distribution.png'))
    plt.close()
    
    # 2. Biểu đồ commit theo giờ trong ngày
    plt.figure(figsize=(12, 6))
    hours = list(range(24))
    commit_counts = [stats['time_analysis']['by_hour'].get(h, 0) for h in hours]
    plt.bar(hours, commit_counts, color='lightgreen')
    plt.title('Commit theo giờ trong ngày')
    plt.xlabel('Giờ (0-23)')
    plt.ylabel('Số lượng commit')
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_commits_by_hour.png'))
    plt.close()
    
    # 3. Biểu đồ commit theo ngày trong tuần
    plt.figure(figsize=(10, 6))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = [stats['time_analysis']['by_day'].get(day, 0) for day in days]
    plt.bar(days, day_counts, color='salmon')
    plt.title('Commit theo ngày trong tuần')
    plt.ylabel('Số lượng commit')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_commits_by_day.png'))
    plt.close()
    
    # 4. Biểu đồ top 10 repository
    plt.figure(figsize=(12, 6))
    top_repos = list(stats['repository_analysis']['commits_per_repo'].items())[:10]
    repo_names = [repo.split('/')[-1] for repo, _ in top_repos]  # Chỉ lấy tên repo, không lấy owner
    repo_counts = [count for _, count in top_repos]
    plt.bar(repo_names, repo_counts, color='lightblue')
    plt.title('Top 10 Repository theo số lượng commit')
    plt.ylabel('Số lượng commit')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_top_repositories.png'))
    plt.close()
    
    # 5. Biểu đồ top 10 ngôn ngữ lập trình
    plt.figure(figsize=(12, 6))
    top_languages = list(stats['file_type_analysis']['languages'].items())[:10]
    lang_names = [lang for lang, _ in top_languages]
    lang_counts = [count for _, count in top_languages]
    plt.bar(lang_names, lang_counts, color='lightcoral')
    plt.title('Top 10 Ngôn ngữ lập trình')
    plt.ylabel('Số lượng file')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_top_languages.png'))
    plt.close()
    
    # 6. Biểu đồ phân bố kích thước thay đổi
    plt.figure(figsize=(10, 6))
    labels = list(stats['changes_analysis']['changes_distribution'].keys())
    values = list(stats['changes_analysis']['changes_distribution'].values())
    plt.bar(labels, values, color='mediumseagreen')
    plt.title('Phân bố kích thước thay đổi trong commit')
    plt.ylabel('Số lượng commit')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_change_size_distribution.png'))
    plt.close()
    
    # 7. Biểu đồ phân bố số lượng file thay đổi
    plt.figure(figsize=(10, 6))
    labels = list(stats['changes_analysis']['files_distribution'].keys())
    values = list(stats['changes_analysis']['files_distribution'].values())
    plt.bar(labels, values, color='mediumpurple')
    plt.title('Phân bố số lượng file thay đổi trong commit')
    plt.ylabel('Số lượng commit')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_files_changed_distribution.png'))
    plt.close()
    
    # 8. Biểu đồ tròn chất lượng commit
    plt.figure(figsize=(10, 6))
    labels = ['Chất lượng cao', 'Chất lượng thấp', 'Chỉ một từ']
    values = [
        stats['quality_analysis']['high_quality_count'], 
        stats['quality_analysis']['low_quality_count'] - stats['quality_analysis']['single_word_count'],
        stats['quality_analysis']['single_word_count']
    ]
    colors = ['#66b3ff', '#ff9999', '#ffcc99']
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Phân bố chất lượng commit message')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_commit_quality_pie.png'))
    plt.close()
    
    logger.info(f"Đã tạo các biểu đồ trực quan trong thư mục {output_dir}")

def save_stats_to_json(stats: Dict, output_file: str) -> None:
    """
    Lưu thống kê vào file JSON.
    
    Args:
        stats: Thống kê đã được phân tích
        output_file: Đường dẫn file đầu ra
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Đã lưu thống kê vào {output_file}")

def generate_html_report(stats: Dict, visualizations_dir: str, output_file: str) -> None:
    """
    Tạo báo cáo HTML từ thống kê và biểu đồ.
    
    Args:
        stats: Thống kê đã được phân tích
        visualizations_dir: Thư mục chứa biểu đồ
        output_file: Đường dẫn file đầu ra
    """
    # Chuyển đổi đường dẫn tuyệt đối thành đường dẫn tương đối
    vis_rel_path = os.path.relpath(visualizations_dir, os.path.dirname(output_file))
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Phân tích dữ liệu Commit GitHub</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .stats-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                min-width: 200px;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .stat-label {{
                font-size: 14px;
                color: #7f8c8d;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
                border-radius: 5px;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 12px;
                color: #7f8c8d;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Báo cáo Phân tích dữ liệu Commit GitHub</h1>
        <p>Báo cáo được tạo vào: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Tổng quan</h2>
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{stats['total_commits']}</div>
                    <div class="stat-label">Tổng số commit</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['repository_analysis']['total_repos']}</div>
                    <div class="stat-label">Số lượng repository</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['author_analysis']['total_authors']}</div>
                    <div class="stat-label">Số lượng tác giả</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['quality_analysis']['high_quality_percentage']:.1f}%</div>
                    <div class="stat-label">Commit chất lượng cao</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Phân bố chất lượng commit message</h3>
                <img src="{vis_rel_path}/08_commit_quality_pie.png" alt="Phân bố chất lượng commit">
            </div>
        </div>
        
        <div class="section">
            <h2>Phân tích Commit Message</h2>
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{stats['commit_length']['mean']:.1f}</div>
                    <div class="stat-label">Độ dài trung bình (từ)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['commit_length']['median']:.0f}</div>
                    <div class="stat-label">Độ dài trung vị (từ)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['commit_length']['max']}</div>
                    <div class="stat-label">Độ dài tối đa (từ)</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Phân bố độ dài commit message</h3>
                <img src="{vis_rel_path}/01_commit_length_distribution.png" alt="Phân bố độ dài commit">
            </div>
        </div>
        
        <div class="section">
            <h2>Phân tích thời gian</h2>
            <div class="chart-container">
                <h3>Commit theo giờ trong ngày</h3>
                <img src="{vis_rel_path}/02_commits_by_hour.png" alt="Commit theo giờ">
            </div>
            
            <div class="chart-container">
                <h3>Commit theo ngày trong tuần</h3>
                <img src="{vis_rel_path}/03_commits_by_day.png" alt="Commit theo ngày">
            </div>
        </div>
        
        <div class="section">
            <h2>Phân tích Repository</h2>
            <div class="chart-container">
                <h3>Top 10 Repository theo số lượng commit</h3>
                <img src="{vis_rel_path}/04_top_repositories.png" alt="Top repositories">
            </div>
            
            <h3>Top 10 Repository</h3>
            <table>
                <tr>
                    <th>Repository</th>
                    <th>Số lượng commit</th>
                </tr>
                {"".join(f"<tr><td>{repo}</td><td>{count}</td></tr>" for repo, count in list(stats['repository_analysis']['commits_per_repo'].items())[:10])}
            </table>
        </div>
        
        <div class="section">
            <h2>Phân tích ngôn ngữ và loại file</h2>
            <div class="chart-container">
                <h3>Top 10 Ngôn ngữ lập trình</h3>
                <img src="{vis_rel_path}/05_top_languages.png" alt="Top languages">
            </div>
            
            <h3>Top 10 Ngôn ngữ lập trình</h3>
            <table>
                <tr>
                    <th>Ngôn ngữ</th>
                    <th>Số lượng file</th>
                </tr>
                {"".join(f"<tr><td>{lang}</td><td>{count}</td></tr>" for lang, count in list(stats['file_type_analysis']['languages'].items())[:10])}
            </table>
        </div>
        
        <div class="section">
            <h2>Phân tích thay đổi</h2>
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{stats['changes_analysis']['mean_changes']:.1f}</div>
                    <div class="stat-label">Số dòng thay đổi trung bình</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['changes_analysis']['median_files']:.0f}</div>
                    <div class="stat-label">Số file thay đổi trung vị</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['changes_analysis']['max_changes']}</div>
                    <div class="stat-label">Số dòng thay đổi tối đa</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Phân bố kích thước thay đổi</h3>
                <img src="{vis_rel_path}/06_change_size_distribution.png" alt="Phân bố kích thước thay đổi">
            </div>
            
            <div class="chart-container">
                <h3>Phân bố số lượng file thay đổi</h3>
                <img src="{vis_rel_path}/07_files_changed_distribution.png" alt="Phân bố số lượng file thay đổi">
            </div>
        </div>
        
        <div class="section">
            <h2>Phân tích tác giả</h2>
            <h3>Top 10 tác giả</h3>
            <table>
                <tr>
                    <th>Tác giả</th>
                    <th>Số lượng commit</th>
                </tr>
                {"".join(f"<tr><td>{author}</td><td>{count}</td></tr>" for author, count in list(stats['author_analysis']['top_authors'].items())[:10])}
            </table>
        </div>
        
        <div class="footer">
            <p>Báo cáo được tạo tự động bởi script phân tích commit GitHub.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Đã tạo báo cáo HTML: {output_file}")

def analyze_commit_data_detailed(input_file: str, output_dir: str, create_visualizations: bool = True, 
                                create_html: bool = True) -> Dict:
    """
    Phân tích chi tiết dữ liệu commit và tạo báo cáo.
    
    Args:
        input_file: Đường dẫn đến file dữ liệu
        output_dir: Thư mục đầu ra cho kết quả phân tích
        create_visualizations: Có tạo biểu đồ trực quan không
        create_html: Có tạo báo cáo HTML không
        
    Returns:
        Thống kê đã được phân tích
    """
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc dữ liệu commit
    commits, metadata = load_commit_data(input_file)
    
    if not commits:
        logger.error("Không có dữ liệu commit để phân tích")
        return {}
    
    logger.info("Bắt đầu phân tích chi tiết...")
    
    # Phân tích các khía cạnh khác nhau
    commit_length_stats = analyze_commit_length(commits)
    time_stats = analyze_commits_by_time(commits)
    changes_stats = analyze_commit_changes(commits)
    repo_stats = analyze_repositories(commits)
    file_type_stats = analyze_file_types(commits)
    merge_stats = analyze_merge_commits(commits)
    author_stats = analyze_authors(commits)
    quality_stats = analyze_commit_quality(commits)
    
    # Tổng hợp thống kê
    stats = {
        'total_commits': len(commits),
        'metadata': metadata,
        'commit_length': commit_length_stats,
        'time_analysis': time_stats,
        'changes_analysis': changes_stats,
        'repository_analysis': repo_stats,
        'file_type_analysis': file_type_stats,
        'merge_analysis': merge_stats,
        'author_analysis': author_stats,
        'quality_analysis': quality_stats,
        'analyzed_at': datetime.now().isoformat()
    }
    
    # Lưu thống kê vào file JSON
    stats_file = os.path.join(output_dir, 'commit_analysis_stats.json')
    save_stats_to_json(stats, stats_file)
    
    # Tạo biểu đồ trực quan
    if create_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        generate_visualizations(stats, vis_dir)
    
    # Tạo báo cáo HTML
    if create_html and create_visualizations:
        html_file = os.path.join(output_dir, 'commit_analysis_report.html')
        generate_html_report(stats, os.path.join(output_dir, 'visualizations'), html_file)
    
    logger.info(f"Đã hoàn thành phân tích chi tiết. Kết quả được lưu vào {output_dir}")
    return stats

def main():
    parser = argparse.ArgumentParser(description='Phân tích chi tiết dữ liệu commit GitHub')
    parser.add_argument('--input', required=True, help='Đường dẫn đến file dữ liệu commit JSON')
    parser.add_argument('--output', default='analysis_results', help='Thư mục đầu ra cho kết quả phân tích')
    parser.add_argument('--no-vis', action='store_true', help='Không tạo biểu đồ trực quan')
    parser.add_argument('--no-html', action='store_true', help='Không tạo báo cáo HTML')
    
    args = parser.parse_args()
    
    analyze_commit_data_detailed(
        input_file=args.input,
        output_dir=args.output,
        create_visualizations=not args.no_vis,
        create_html=not args.no_html
    )

if __name__ == "__main__":
    main()
