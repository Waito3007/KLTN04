"""
Script để tải dataset commit từ Kaggle và chuẩn bị dữ liệu cho mô hình HAN
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import zipfile
import shutil
from typing import List, Dict, Any, Tuple
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

class KaggleDatasetDownloader:
    def __init__(self, base_dir: str = None):
        """
        Khởi tạo class để tải dataset từ Kaggle
        
        Args:
            base_dir: Thư mục gốc để lưu dữ liệu
        """
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.data_dir = os.path.join(self.base_dir, 'kaggle_data')
        self.processed_dir = os.path.join(self.base_dir, 'training_data')
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def check_kaggle_config(self) -> bool:
        """Kiểm tra cấu hình Kaggle API"""
        try:
            import kaggle
            logger.info("✅ Kaggle API đã được cấu hình")
            return True
        except ImportError:
            logger.error("❌ Kaggle package chưa được cài đặt. Chạy: pip install kaggle")
            return False
        except OSError as e:
            logger.error(f"❌ Lỗi cấu hình Kaggle API: {e}")
            logger.info("Vui lòng:")
            logger.info("1. Tạo API token tại: https://www.kaggle.com/settings")
            logger.info("2. Đặt file kaggle.json vào ~/.kaggle/ (Linux/Mac) hoặc C:\\Users\\<username>\\.kaggle\\ (Windows)")
            logger.info("3. Cấp quyền 600 cho file: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """
        Tải dataset từ Kaggle
        
        Args:
            dataset_name: Tên dataset trên Kaggle (format: username/dataset-name)
            force_download: Có tải lại nếu đã tồn tại hay không
            
        Returns:
            bool: True nếu thành công
        """
        if not self.check_kaggle_config():
            return False
            
        try:
            import kaggle
            
            dataset_path = os.path.join(self.data_dir, dataset_name.split('/')[-1])
            
            if os.path.exists(dataset_path) and not force_download:
                logger.info(f"Dataset {dataset_name} đã tồn tại, bỏ qua tải xuống")
                return True
                
            logger.info(f"🔄 Đang tải dataset: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=self.data_dir, 
                unzip=True
            )
            
            logger.info(f"✅ Tải thành công dataset: {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải dataset {dataset_name}: {e}")
            return False
    
    def list_popular_commit_datasets(self) -> List[str]:
        """Liệt kê các dataset commit phổ biến trên Kaggle"""
        return [
            "shashankbansal6/git-commits-message-dataset",
            "madhav28/git-commit-messages",
            "aashita/git-commit-messages",
            "jainaru/commit-classification-dataset",
            "shubhamjain0594/commit-message-generation",
            "saurabhshahane/conventional-commit-messages",
            "devanshunigam/commits",
            "ashydv/commits-dataset"
        ]
    
    def process_commit_dataset(self, csv_files: List[str]) -> Dict[str, Any]:
        """
        Xử lý dữ liệu commit từ các file CSV
        
        Args:
            csv_files: Danh sách các file CSV
            
        Returns:
            Dict chứa dữ liệu đã xử lý
        """
        all_data = []
        
        for csv_file in csv_files:
            logger.info(f"🔄 Đang xử lý file: {csv_file}")
            
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"📊 Số lượng records: {len(df)}")
                logger.info(f"📋 Các cột: {list(df.columns)}")
                
                # Chuẩn hóa tên cột
                df.columns = df.columns.str.lower().str.strip()
                
                # Tìm cột chứa commit message
                message_cols = [col for col in df.columns if 
                              any(keyword in col for keyword in ['message', 'commit', 'msg', 'text', 'description'])]
                
                if not message_cols:
                    logger.warning(f"⚠️ Không tìm thấy cột commit message trong {csv_file}")
                    continue
                
                message_col = message_cols[0]
                logger.info(f"📝 Sử dụng cột '{message_col}' làm commit message")
                
                # Xử lý dữ liệu
                for _, row in df.iterrows():
                    commit_msg = str(row.get(message_col, '')).strip()
                    
                    if not commit_msg or commit_msg == 'nan' or len(commit_msg) < 5:
                        continue
                    
                    # Trích xuất thông tin khác nếu có
                    author = str(row.get('author', row.get('committer', 'unknown'))).strip()
                    repo = str(row.get('repo', row.get('repository', row.get('project', 'unknown')))).strip()
                    
                    # Phân loại commit dựa trên message
                    commit_type = self.classify_commit_type(commit_msg)
                    purpose = self.classify_commit_purpose(commit_msg)
                    sentiment = self.classify_sentiment(commit_msg)
                    tech_tags = self.extract_tech_tags(commit_msg)
                    
                    data_point = {
                        'commit_message': commit_msg,
                        'commit_type': commit_type,
                        'purpose': purpose,
                        'sentiment': sentiment,
                        'tech_tag': tech_tags[0] if tech_tags else 'general',
                        'author': author if author != 'nan' else 'unknown',
                        'source_repo': repo if repo != 'nan' else 'unknown'
                    }
                    
                    all_data.append(data_point)
                    
            except Exception as e:
                logger.error(f"❌ Lỗi khi xử lý file {csv_file}: {e}")
                continue
        
        logger.info(f"✅ Tổng cộng xử lý được {len(all_data)} commit messages")
        return {'data': all_data, 'total_count': len(all_data)}
    
    def classify_commit_type(self, message: str) -> str:
        """Phân loại loại commit dựa trên message"""
        message_lower = message.lower()
        
        # Conventional commit patterns
        if message_lower.startswith(('feat:', 'feature:')):return 'feat'
        elif message_lower.startswith(('fix:', 'bugfix:')):return 'fix'
        elif message_lower.startswith(('docs:', 'doc:')):return 'docs'
        elif message_lower.startswith(('style:', 'format:')):return 'style'
        elif message_lower.startswith(('refactor:', 'refact:')):return 'refactor'
        elif message_lower.startswith(('test:', 'tests:')):return 'test'
        elif message_lower.startswith(('chore:', 'build:', 'ci:')):return 'chore'
        
        # Keyword-based classification
        elif any(word in message_lower for word in ['add', 'implement', 'create', 'new']):
            return 'feat'
        elif any(word in message_lower for word in ['fix', 'bug', 'error', 'issue']):
            return 'fix'
        elif any(word in message_lower for word in ['update', 'modify', 'change']):
            return 'feat'
        elif any(word in message_lower for word in ['remove', 'delete', 'clean']):
            return 'chore'
        elif any(word in message_lower for word in ['test', 'spec']):
            return 'test'
        elif any(word in message_lower for word in ['doc', 'readme', 'comment']):
            return 'docs'
        else:
            return 'other'
    
    def classify_commit_purpose(self, message: str) -> str:
        """Phân loại mục đích commit"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['feature', 'feat', 'add', 'implement', 'new']):
            return 'Feature Implementation'
        elif any(word in message_lower for word in ['fix', 'bug', 'error', 'issue', 'patch']):
            return 'Bug Fix'
        elif any(word in message_lower for word in ['refactor', 'restructure', 'reorganize']):
            return 'Refactoring'
        elif any(word in message_lower for word in ['doc', 'readme', 'comment', 'documentation']):
            return 'Documentation Update'
        elif any(word in message_lower for word in ['test', 'spec', 'testing']):
            return 'Test Update'
        elif any(word in message_lower for word in ['security', 'secure', 'vulnerability']):
            return 'Security Patch'
        elif any(word in message_lower for word in ['style', 'format', 'lint', 'prettier']):
            return 'Code Style/Formatting'
        elif any(word in message_lower for word in ['build', 'ci', 'cd', 'deploy', 'pipeline']):
            return 'Build/CI/CD Script Update'
        else:
            return 'Other'
    
    def classify_sentiment(self, message: str) -> str:
        """Phân loại cảm xúc trong commit message"""
        message_lower = message.lower()
        
        positive_words = ['improve', 'enhance', 'optimize', 'upgrade', 'better', 'good', 'great', 'awesome']
        negative_words = ['bug', 'error', 'issue', 'problem', 'fail', 'broken', 'wrong']
        urgent_words = ['urgent', 'critical', 'hotfix', 'emergency', 'asap']
        
        if any(word in message_lower for word in urgent_words):
            return 'urgent'
        elif any(word in message_lower for word in positive_words):
            return 'positive'
        elif any(word in message_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'
    
    def extract_tech_tags(self, message: str) -> List[str]:
        """Trích xuất các tag công nghệ từ commit message"""
        message_lower = message.lower()
        tech_tags = []
        
        tech_keywords = {
            'javascript': ['js', 'javascript', 'node', 'npm', 'yarn'],
            'python': ['python', 'py', 'pip', 'django', 'flask'],
            'java': ['java', 'maven', 'gradle', 'spring'],
            'react': ['react', 'jsx', 'component'],
            'vue': ['vue', 'vuex', 'nuxt'],
            'angular': ['angular', 'ng', 'typescript'],
            'css': ['css', 'sass', 'scss', 'less', 'style'],
            'html': ['html', 'dom', 'markup'],
            'database': ['sql', 'mysql', 'postgres', 'mongodb', 'database', 'db'],
            'api': ['api', 'rest', 'graphql', 'endpoint'],
            'docker': ['docker', 'container', 'dockerfile'],
            'git': ['git', 'merge', 'branch', 'commit'],
            'testing': ['test', 'spec', 'jest', 'mocha', 'junit'],
            'security': ['security', 'auth', 'oauth', 'jwt', 'ssl'],
            'performance': ['performance', 'optimize', 'cache', 'speed'],
            'ui': ['ui', 'ux', 'interface', 'design', 'layout']
        }
        
        for category, keywords in tech_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                tech_tags.append(category)
        
        return tech_tags if tech_tags else ['general']
    
    def save_processed_data(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        Lưu dữ liệu đã xử lý theo định dạng cho HAN model
        
        Args:
            data: Dữ liệu đã xử lý
            filename: Tên file để lưu
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'kaggle_training_data_{timestamp}.json'
        
        filepath = os.path.join(self.processed_dir, filename)
        
        # Chuẩn bị dữ liệu theo format HAN
        han_format_data = []
        
        for item in data['data']:
            han_item = {
                'text': item['commit_message'],
                'labels': {
                    'commit_type': item['commit_type'],
                    'purpose': item['purpose'],
                    'sentiment': item['sentiment'],
                    'tech_tag': item['tech_tag'],
                    'author': item['author'],
                    'source_repo': item['source_repo']
                }
            }
            han_format_data.append(han_item)
        
        # Thống kê dữ liệu
        stats = self.generate_statistics(han_format_data)
        
        # Lưu file
        output_data = {
            'metadata': {
                'total_samples': len(han_format_data),
                'created_at': datetime.now().isoformat(),
                'source': 'kaggle_datasets',
                'statistics': stats
            },
            'data': han_format_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Đã lưu {len(han_format_data)} samples vào {filepath}")
        return filepath
    
    def generate_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """Tạo thống kê cho dữ liệu"""
        stats = {}
        
        # Đếm theo từng label category
        for label_type in ['commit_type', 'purpose', 'sentiment', 'tech_tag']:
            label_counts = {}
            for item in data:
                label = item['labels'][label_type]
                label_counts[label] = label_counts.get(label, 0) + 1
            stats[label_type] = label_counts
        
        # Thống kê độ dài text
        text_lengths = [len(item['text'].split()) for item in data]
        stats['text_length'] = {
            'min': min(text_lengths),
            'max': max(text_lengths),
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths)
        }
        
        return stats
    
    def download_and_process_datasets(self, dataset_names: List[str] = None) -> List[str]:
        """
        Tải và xử lý nhiều dataset cùng lúc
        
        Args:
            dataset_names: Danh sách tên dataset, nếu None sẽ dùng danh sách mặc định
            
        Returns:
            List[str]: Danh sách đường dẫn file đã xử lý
        """
        if not dataset_names:
            dataset_names = self.list_popular_commit_datasets()
        
        processed_files = []
        
        logger.info(f"🎯 Bắt đầu tải và xử lý {len(dataset_names)} datasets")
        
        for i, dataset_name in enumerate(dataset_names, 1):
            logger.info(f"\n📦 [{i}/{len(dataset_names)}] Xử lý dataset: {dataset_name}")
            
            # Tải dataset
            if not self.download_dataset(dataset_name):
                logger.warning(f"⚠️ Bỏ qua dataset {dataset_name} do lỗi tải xuống")
                continue
            
            # Tìm file CSV trong thư mục dataset
            dataset_dir = os.path.join(self.data_dir)
            csv_files = []
            
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                logger.warning(f"⚠️ Không tìm thấy file CSV trong dataset {dataset_name}")
                continue
            
            # Xử lý dữ liệu
            try:
                processed_data = self.process_commit_dataset(csv_files)
                
                if processed_data['total_count'] > 0:
                    # Lưu dữ liệu với tên dataset
                    dataset_short_name = dataset_name.split('/')[-1].replace('-', '_')
                    filename = f'kaggle_{dataset_short_name}_{datetime.now().strftime("%Y%m%d")}.json'
                    
                    saved_file = self.save_processed_data(processed_data, filename)
                    processed_files.append(saved_file)
                else:
                    logger.warning(f"⚠️ Không có dữ liệu hợp lệ từ dataset {dataset_name}")
                    
            except Exception as e:
                logger.error(f"❌ Lỗi khi xử lý dataset {dataset_name}: {e}")
                continue
        
        logger.info(f"\n🎉 Hoàn thành! Đã xử lý {len(processed_files)} datasets thành công")
        return processed_files
    
    def merge_datasets(self, json_files: List[str], output_filename: str = None) -> str:
        """
        Gộp nhiều file JSON thành một file duy nhất
        
        Args:
            json_files: Danh sách đường dẫn file JSON
            output_filename: Tên file output
            
        Returns:
            str: Đường dẫn file đã gộp
        """
        if not output_filename:
            output_filename = f'merged_kaggle_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        output_path = os.path.join(self.processed_dir, output_filename)
        
        all_data = []
        total_stats = {}
        
        logger.info(f"🔄 Gộp {len(json_files)} files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    all_data.extend(file_data['data'])
                    
                    # Gộp thống kê
                    if 'statistics' in file_data.get('metadata', {}):
                        file_stats = file_data['metadata']['statistics']
                        for key, value in file_stats.items():
                            if key not in total_stats:
                                total_stats[key] = {}
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if subkey in total_stats[key]:
                                        total_stats[key][subkey] += subvalue
                                    else:
                                        total_stats[key][subkey] = subvalue
                        
            except Exception as e:
                logger.error(f"❌ Lỗi khi đọc file {json_file}: {e}")
                continue
        
        # Lưu file gộp
        merged_data = {
            'metadata': {
                'total_samples': len(all_data),
                'created_at': datetime.now().isoformat(),
                'source': 'merged_kaggle_datasets',
                'source_files': json_files,
                'statistics': total_stats
            },
            'data': all_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Đã gộp {len(all_data)} samples vào {output_path}")
        return output_path


def main():
    """Hàm chính để chạy script"""
    print("=" * 80)
    print("🚀 KAGGLE DATASET DOWNLOADER VÀ PROCESSOR CHO HAN MODEL")
    print("=" * 80)
    
    # Khởi tạo downloader
    downloader = KaggleDatasetDownloader()
    
    # Hiển thị menu
    print("\n📋 Các tùy chọn:")
    print("1. Tải và xử lý tất cả datasets phổ biến")
    print("2. Tải và xử lý dataset cụ thể")
    print("3. Chỉ xử lý dữ liệu có sẵn")
    print("4. Hiển thị danh sách datasets phổ biến")
    
    choice = input("\n🔸 Chọn tùy chọn (1-4): ").strip()
    
    if choice == '1':
        # Tải tất cả datasets phổ biến
        logger.info("📦 Tải tất cả datasets phổ biến...")
        processed_files = downloader.download_and_process_datasets()
        
        if processed_files:
            # Gộp tất cả files
            if len(processed_files) > 1:
                merged_file = downloader.merge_datasets(processed_files)
                logger.info(f"🎯 File dữ liệu cuối cùng: {merged_file}")
            else:
                logger.info(f"🎯 File dữ liệu: {processed_files[0]}")
        
    elif choice == '2':
        # Tải dataset cụ thể
        dataset_name = input("🔸 Nhập tên dataset (format: username/dataset-name): ").strip()
        if dataset_name:
            processed_files = downloader.download_and_process_datasets([dataset_name])
            if processed_files:
                logger.info(f"🎯 File dữ liệu: {processed_files[0]}")
        
    elif choice == '3':
        # Xử lý dữ liệu có sẵn
        csv_files = []
        for root, dirs, files in os.walk(downloader.data_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            logger.info(f"🔍 Tìm thấy {len(csv_files)} file CSV")
            processed_data = downloader.process_commit_dataset(csv_files)
            if processed_data['total_count'] > 0:
                saved_file = downloader.save_processed_data(processed_data)
                logger.info(f"🎯 File dữ liệu: {saved_file}")
        else:
            logger.warning("❌ Không tìm thấy file CSV nào")
        
    elif choice == '4':
        # Hiển thị danh sách
        datasets = downloader.list_popular_commit_datasets()
        print("\n📋 Danh sách datasets commit phổ biến:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")
    
    else:
        logger.error("❌ Lựa chọn không hợp lệ")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH!")
    print("=" * 80)


if __name__ == "__main__":
    main()
