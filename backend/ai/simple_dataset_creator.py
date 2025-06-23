"""
Alternative Kaggle Dataset Downloader
Tải dataset từ Kaggle mà không cần API (sử dụng public URLs)
"""

import os
import requests
import zipfile
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import re

class SimpleKaggleDownloader:
    def __init__(self, data_dir="kaggle_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_file(self, url, filename):
        """Download file từ URL"""
        try:
            print(f"📥 Đang tải {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Đã tải: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Lỗi tải {filename}: {str(e)}")
            return None
    
    def extract_zip(self, zip_path):
        """Giải nén file zip"""
        try:
            extract_dir = zip_path.parent / zip_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"📂 Đã giải nén: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            print(f"❌ Lỗi giải nén: {str(e)}")
            return None
    
    def create_sample_commit_data(self):
        """Tạo dữ liệu commit mẫu cho testing"""
        print("🎯 Tạo dữ liệu commit mẫu...")
        
        sample_commits = [
            {
                "message": "feat: add user authentication with JWT tokens",
                "author": "john_doe",
                "files_changed": 5,
                "insertions": 120,
                "deletions": 15,
                "repo": "webapp"
            },
            {
                "message": "fix: resolve memory leak in data processing module",
                "author": "jane_smith", 
                "files_changed": 2,
                "insertions": 25,
                "deletions": 40,
                "repo": "backend"
            },
            {
                "message": "docs: update API documentation for v2.0",
                "author": "dev_team",
                "files_changed": 8,
                "insertions": 200,
                "deletions": 50,
                "repo": "docs"
            },
            {
                "message": "refactor: optimize database queries and connection pool",
                "author": "db_admin",
                "files_changed": 4,
                "insertions": 80,
                "deletions": 120,
                "repo": "backend"
            },
            {
                "message": "test: add comprehensive unit tests for user service",
                "author": "qa_engineer",
                "files_changed": 6,
                "insertions": 300,
                "deletions": 10,
                "repo": "backend"
            },
            {
                "message": "style: format code and fix ESLint warnings",
                "author": "formatter_bot",
                "files_changed": 15,
                "insertions": 50,
                "deletions": 60,
                "repo": "frontend"
            },
            {
                "message": "chore: update dependencies and build configuration",
                "author": "maintainer",
                "files_changed": 3,
                "insertions": 20,
                "deletions": 25,
                "repo": "config"
            },
            {
                "message": "feat(ui): implement responsive dashboard layout",
                "author": "ui_designer",
                "files_changed": 10,
                "insertions": 400,
                "deletions": 100,
                "repo": "frontend"
            },
            {
                "message": "fix(security): patch SQL injection vulnerability",
                "author": "security_team",
                "files_changed": 3,
                "insertions": 45,
                "deletions": 20,
                "repo": "backend"
            },
            {
                "message": "perf: improve loading time by 50% with caching",
                "author": "performance_team",
                "files_changed": 7,
                "insertions": 150,
                "deletions": 80,
                "repo": "backend"
            }
        ]
        
        # Mở rộng dataset với variations
        extended_commits = []
        variations = [
            "Add {feature} functionality to {component}",
            "Fix {issue} in {module} component", 
            "Update {item} for better {aspect}",
            "Refactor {code_part} for improved {quality}",
            "Remove deprecated {old_feature} from {location}",
            "Implement {new_feature} with {technology}",
            "Optimize {process} performance in {area}",
            "Configure {tool} for {purpose}",
            "Integrate {service} with {system}",
            "Enhance {feature} with {improvement}"
        ]
        
        features = ["authentication", "validation", "caching", "logging", "monitoring"]
        components = ["user interface", "API endpoints", "database layer", "frontend", "backend"]
        issues = ["memory leak", "race condition", "null pointer", "buffer overflow", "timeout"]
        modules = ["payment", "user management", "data processing", "file upload", "notification"]
        
        for i, template in enumerate(variations):
            for j in range(10):  # 10 variations per template
                message = template.format(
                    feature=features[j % len(features)],
                    component=components[j % len(components)],
                    issue=issues[j % len(issues)],
                    module=modules[j % len(modules)],
                    item=f"configuration {j}",
                    aspect="performance",
                    code_part="utility functions",
                    quality="maintainability",
                    old_feature=f"legacy feature {j}",
                    location="main module",
                    new_feature=f"feature {j}",
                    technology="modern framework",
                    process="data processing",
                    area="core system",
                    tool="build tool",
                    purpose="automation",
                    service="external API",
                    system="main application",
                    improvement="better UX"
                )
                
                extended_commits.append({
                    "message": message,
                    "author": f"developer_{(i*10 + j) % 20}",
                    "files_changed": (j % 10) + 1,
                    "insertions": (j * 20) + 50,
                    "deletions": (j * 10) + 10,
                    "repo": ["frontend", "backend", "mobile", "api", "database"][j % 5]
                })
        
        all_commits = sample_commits + extended_commits
        
        # Lưu thành CSV
        df = pd.DataFrame(all_commits)
        csv_path = self.data_dir / "sample_commits.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Đã tạo {len(all_commits)} commit samples: {csv_path}")
        return csv_path
    
    def process_commit_data(self, csv_file):
        """Xử lý dữ liệu commit thành format phù hợp với HAN"""
        print(f"🔄 Đang xử lý dữ liệu từ {csv_file}...")
        
        df = pd.read_csv(csv_file)
        processed_data = []
        
        def classify_commit_type(message):
            """Phân loại commit type từ message"""
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['feat', 'feature', 'add', 'implement', 'new']):
                return 'feat'
            elif any(word in message_lower for word in ['fix', 'bug', 'resolve', 'patch']):
                return 'fix'
            elif any(word in message_lower for word in ['doc', 'readme', 'comment', 'documentation']):
                return 'docs'
            elif any(word in message_lower for word in ['style', 'format', 'lint', 'prettier']):
                return 'style'  
            elif any(word in message_lower for word in ['refactor', 'restructure', 'reorganize']):
                return 'refactor'
            elif any(word in message_lower for word in ['test', 'spec', 'unittest']):
                return 'test'
            elif any(word in message_lower for word in ['chore', 'update', 'config', 'build']):
                return 'chore'
            else:
                return 'other'
        
        def classify_purpose(message):
            """Phân loại mục đích từ message"""
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['feat', 'feature', 'add', 'implement', 'new']):
                return 'Feature Implementation'
            elif any(word in message_lower for word in ['fix', 'bug', 'resolve', 'patch']):
                return 'Bug Fix'
            elif any(word in message_lower for word in ['refactor', 'optimize', 'improve']):
                return 'Refactoring'
            elif any(word in message_lower for word in ['doc', 'readme', 'comment']):
                return 'Documentation Update'
            elif any(word in message_lower for word in ['test', 'spec', 'unittest']):
                return 'Test Update'
            elif any(word in message_lower for word in ['security', 'vulnerability', 'patch']):
                return 'Security Patch'
            elif any(word in message_lower for word in ['style', 'format', 'lint']):
                return 'Code Style/Formatting'
            elif any(word in message_lower for word in ['build', 'ci', 'cd', 'deploy']):
                return 'Build/CI/CD Script Update'
            else:
                return 'Other'
        
        def classify_sentiment(message):
            """Phân loại sentiment"""
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['critical', 'urgent', 'hotfix', 'emergency']):
                return 'urgent'
            elif any(word in message_lower for word in ['improve', 'enhance', 'optimize', 'better']):
                return 'positive'
            elif any(word in message_lower for word in ['fix', 'bug', 'error', 'issue', 'problem']):
                return 'negative'
            else:
                return 'neutral'
        
        def classify_tech_tag(message, repo):
            """Phân loại tech tag"""
            message_lower = message.lower()
            repo_lower = repo.lower() if pd.notna(repo) else ''
            
            combined = f"{message_lower} {repo_lower}"
            
            if any(word in combined for word in ['js', 'javascript', 'react', 'vue', 'angular', 'node']):
                return 'javascript'
            elif any(word in combined for word in ['py', 'python', 'django', 'flask']):
                return 'python'
            elif any(word in combined for word in ['java', 'spring', 'maven']):
                return 'java'
            elif any(word in combined for word in ['css', 'sass', 'scss', 'style']):
                return 'css'
            elif any(word in combined for word in ['html', 'template', 'markup']):
                return 'html'
            elif any(word in combined for word in ['database', 'sql', 'mysql', 'postgres']):
                return 'database'
            elif any(word in combined for word in ['api', 'rest', 'graphql', 'endpoint']):
                return 'api'
            elif any(word in combined for word in ['docker', 'container', 'k8s']):
                return 'docker'
            elif any(word in combined for word in ['git', 'commit', 'merge', 'branch']):
                return 'git'
            elif any(word in combined for word in ['test', 'spec', 'unittest']):
                return 'testing'
            elif any(word in combined for word in ['security', 'auth', 'ssl', 'encrypt']):
                return 'security'
            elif any(word in combined for word in ['performance', 'optimize', 'cache']):
                return 'performance'
            elif any(word in combined for word in ['ui', 'ux', 'interface', 'frontend']):
                return 'ui'
            else:
                return 'general'
        
        for _, row in df.iterrows():
            message = str(row['message'])
            repo = str(row.get('repo', ''))
            
            processed_data.append({
                "text": message,
                "labels": {
                    "commit_type": classify_commit_type(message),
                    "purpose": classify_purpose(message),
                    "sentiment": classify_sentiment(message),
                    "tech_tag": classify_tech_tag(message, repo),
                    "author": str(row.get('author', 'unknown')),
                    "source_repo": repo
                }
            })
        
        # Tính thống kê
        all_labels = {
            'commit_type': {},
            'purpose': {},
            'sentiment': {},
            'tech_tag': {}
        }
        
        for item in processed_data:
            for label_type, label_value in item['labels'].items():
                if label_type in all_labels:
                    all_labels[label_type][label_value] = all_labels[label_type].get(label_value, 0) + 1
        
        # Tạo metadata
        metadata = {
            "total_samples": len(processed_data),
            "created_at": datetime.now().isoformat(),
            "source": "sample_data",
            "statistics": all_labels
        }
        
        # Lưu kết quả
        result = {
            "metadata": metadata,
            "data": processed_data
        }
        
        output_file = Path("training_data") / "han_training_samples.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Đã xử lý {len(processed_data)} samples")
        print(f"💾 Dữ liệu đã lưu: {output_file}")
        
        # In thống kê
        print("\n📊 Thống kê nhãn:")
        for label_type, counts in all_labels.items():
            print(f"\n{label_type.upper()}:")
            for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {count}")
        
        return output_file

def main():
    print("🚀 SIMPLE KAGGLE DATASET DOWNLOADER")
    print("="*60)
    print("Tool này tạo dữ liệu mẫu khi không thể kết nối Kaggle API")
    
    # Tạo downloader
    downloader = SimpleKaggleDownloader()
    
    print("\n📋 Các tùy chọn:")
    print("1. Tạo dữ liệu commit mẫu (Khuyên dùng khi test)")
    print("2. Tải từ URL trực tiếp (nếu có)")
    print("3. Xử lý file CSV có sẵn")
    
    choice = input("\nNhập lựa chọn (1-3): ").strip()
    
    if choice == '1':
        # Tạo dữ liệu mẫu
        csv_file = downloader.create_sample_commit_data()
        if csv_file:
            # Xử lý dữ liệu
            training_file = downloader.process_commit_data(csv_file)
            print(f"\n🎉 Hoàn thành! Dữ liệu training: {training_file}")
            print("\n📝 Bước tiếp theo:")
            print("   python train_han_with_kaggle.py")
    
    elif choice == '2':
        url = input("Nhập URL để tải: ").strip()
        if url:
            filename = input("Nhập tên file (hoặc Enter để tự động): ").strip()
            if not filename:
                filename = url.split('/')[-1] or "downloaded_file"
            
            downloaded = downloader.download_file(url, filename)
            if downloaded:
                print(f"✅ Đã tải: {downloaded}")
    
    elif choice == '3':
        csv_file = input("Nhập đường dẫn file CSV: ").strip()
        if os.path.exists(csv_file):
            training_file = downloader.process_commit_data(csv_file)
            print(f"\n🎉 Hoàn thành! Dữ liệu training: {training_file}")
        else:
            print(f"❌ File không tồn tại: {csv_file}")
    
    else:
        print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main()
