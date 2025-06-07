#!/usr/bin/env python3
"""
Script tích hợp để tải dữ liệu từ Kaggle và train mô hình HAN
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import các module cần thiết
from download_kaggle_dataset import KaggleDatasetDownloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HANKaggleTrainer:
    def __init__(self, base_dir: str = None):
        """
        Khởi tạo trainer tích hợp Kaggle + HAN
        
        Args:
            base_dir: Thư mục gốc của project
        """
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.downloader = KaggleDatasetDownloader(self.base_dir)
        
    def prepare_data_from_kaggle(self, dataset_names: list = None, merge_files: bool = True) -> str:
        """
        Tải và chuẩn bị dữ liệu từ Kaggle
        
        Args:
            dataset_names: Danh sách tên dataset
            merge_files: Có gộp files không
            
        Returns:
            str: Đường dẫn file dữ liệu cuối cùng
        """
        logger.info("🔄 Bắt đầu tải dữ liệu từ Kaggle...")
        
        # Tải và xử lý datasets
        processed_files = self.downloader.download_and_process_datasets(dataset_names)
        
        if not processed_files:
            raise Exception("Không thể tải được dữ liệu nào từ Kaggle")
        
        # Gộp files nếu cần
        if merge_files and len(processed_files) > 1:
            logger.info("🔄 Gộp các files dữ liệu...")
            final_file = self.downloader.merge_datasets(processed_files)
        else:
            final_file = processed_files[0]
        
        logger.info(f"✅ Dữ liệu đã sẵn sàng: {final_file}")
        return final_file
    
    def validate_data_for_han(self, data_file: str) -> bool:
        """
        Kiểm tra dữ liệu có phù hợp với HAN model không
        
        Args:
            data_file: Đường dẫn file dữ liệu
            
        Returns:
            bool: True nếu dữ liệu hợp lệ
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Kiểm tra cấu trúc
            if 'data' not in data:
                logger.error("❌ Thiếu trường 'data' trong file")
                return False
            
            if not data['data']:
                logger.error("❌ Không có dữ liệu trong file")
                return False
            
            # Kiểm tra sample đầu tiên
            first_sample = data['data'][0]
            required_fields = ['text', 'labels']
            
            for field in required_fields:
                if field not in first_sample:
                    logger.error(f"❌ Thiếu trường '{field}' trong sample")
                    return False
            
            # Kiểm tra labels
            labels = first_sample['labels']
            required_labels = ['commit_type', 'purpose', 'sentiment', 'tech_tag']
            
            for label in required_labels:
                if label not in labels:
                    logger.warning(f"⚠️ Thiếu label '{label}' - sẽ được thêm giá trị mặc định")
            
            logger.info("✅ Dữ liệu hợp lệ cho HAN model")
            
            # In thống kê
            total_samples = len(data['data'])
            logger.info(f"📊 Tổng số samples: {total_samples}")
            
            if 'metadata' in data and 'statistics' in data['metadata']:
                stats = data['metadata']['statistics']
                logger.info("📊 Thống kê nhãn:")
                for label_type, counts in stats.items():
                    if isinstance(counts, dict):
                        logger.info(f"  {label_type}: {dict(list(counts.items())[:5])}")  # Top 5
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi kiểm tra dữ liệu: {e}")
            return False
    
    def create_han_training_config(self, data_file: str) -> dict:
        """
        Tạo cấu hình training cho HAN model
        
        Args:
            data_file: Đường dẫn file dữ liệu
            
        Returns:
            dict: Cấu hình training
        """
        # Load data để tính toán cấu hình
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_samples = len(data['data'])
        
        # Tính toán batch size và epochs phù hợp
        if total_samples < 1000:
            batch_size = 16
            epochs = 50
        elif total_samples < 10000:
            batch_size = 32
            epochs = 30
        else:
            batch_size = 64
            epochs = 20
        
        config = {
            'data_file': data_file,
            'model_config': {
                'vocab_size': 10000,
                'embedding_dim': 100,
                'hidden_dim': 128,
                'num_classes': {
                    'commit_type': 9,  # feat, fix, docs, style, refactor, test, chore, other
                    'purpose': 9,      # Feature Implementation, Bug Fix, etc.
                    'sentiment': 4,    # positive, negative, neutral, urgent
                    'tech_tag': 16,    # javascript, python, etc.
                },
                'dropout': 0.3
            },
            'training_config': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'val_split': 0.2,
                'early_stopping_patience': 5
            },
            'output_config': {
                'model_dir': os.path.join(self.base_dir, 'models', 'han_kaggle_model'),
                'log_dir': os.path.join(self.base_dir, 'training_logs'),
                'checkpoint_dir': os.path.join(self.base_dir, 'checkpoints')
            }
        }
        
        # Tạo thư mục output
        for dir_path in config['output_config'].values():
            os.makedirs(dir_path, exist_ok=True)
        
        return config
    
    def start_han_training(self, config: dict) -> bool:
        """
        Bắt đầu training HAN model
        
        Args:
            config: Cấu hình training
            
        Returns:
            bool: True nếu training thành công
        """
        try:
            logger.info("🚀 Bắt đầu training HAN model...")
            
            # Import HAN training module
            try:
                from train_han_multitask import main as train_han_main
                
                # Cập nhật config cho HAN trainer
                # Có thể cần modify train_han_multitask.py để chấp nhận config từ bên ngoài
                
                # Tạm thời chạy script trực tiếp
                import subprocess
                
                script_path = os.path.join(self.base_dir, 'train_han_multitask.py')
                result = subprocess.run([
                    sys.executable, script_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Training HAN model thành công!")
                    return True
                else:
                    logger.error(f"❌ Lỗi khi training: {result.stderr}")
                    return False
                    
            except ImportError:
                logger.error("❌ Không thể import HAN training module")
                logger.info("💡 Hãy đảm bảo file train_han_multitask.py tồn tại")
                return False
                
        except Exception as e:
            logger.error(f"❌ Lỗi khi training HAN model: {e}")
            return False
    
    def run_full_pipeline(self, dataset_names: list = None) -> bool:
        """
        Chạy toàn bộ pipeline từ tải dữ liệu đến training
        
        Args:
            dataset_names: Danh sách dataset cần tải
            
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("🎯 BẮT ĐẦU FULL PIPELINE KAGGLE + HAN")
            logger.info("=" * 60)
            
            # Bước 1: Tải dữ liệu từ Kaggle
            logger.info("📦 BƯỚC 1: Tải dữ liệu từ Kaggle")
            data_file = self.prepare_data_from_kaggle(dataset_names)
            
            # Bước 2: Kiểm tra dữ liệu
            logger.info("🔍 BƯỚC 2: Kiểm tra dữ liệu")
            if not self.validate_data_for_han(data_file):
                return False
            
            # Bước 3: Tạo cấu hình training
            logger.info("⚙️ BƯỚC 3: Tạo cấu hình training")
            config = self.create_han_training_config(data_file)
            
            # Lưu cấu hình
            config_file = os.path.join(self.base_dir, f'han_training_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Đã lưu cấu hình: {config_file}")
            
            # Bước 4: Training model
            logger.info("🚀 BƯỚC 4: Training HAN model")
            training_success = self.start_han_training(config)
            
            if training_success:
                logger.info("🎉 PIPELINE HOÀN THÀNH THÀNH CÔNG!")
                logger.info(f"📊 Dữ liệu: {data_file}")
                logger.info(f"⚙️ Cấu hình: {config_file}")
                logger.info(f"🤖 Model: {config['output_config']['model_dir']}")
                return True
            else:
                logger.error("❌ PIPELINE THẤT BẠI Ở BƯỚC TRAINING")
                return False
                
        except Exception as e:
            logger.error(f"❌ LỖI TRONG PIPELINE: {e}")
            return False

def main():
    """Hàm chính"""
    print("🎯 HAN MODEL TRAINING VỚI KAGGLE DATASET")
    print("=" * 80)
    
    trainer = HANKaggleTrainer()
    
    # Menu lựa chọn
    print("\n📋 Các tùy chọn:")
    print("1. Chạy full pipeline (tải dữ liệu + training)")
    print("2. Chỉ tải dữ liệu từ Kaggle")
    print("3. Training với dữ liệu có sẵn")
    print("4. Kiểm tra dữ liệu hiện có")
    
    choice = input("\n🔸 Chọn tùy chọn (1-4): ").strip()
    
    if choice == '1':
        # Full pipeline
        print("\n🔸 Chọn loại dataset:")
        print("1. Tất cả datasets phổ biến")
        print("2. Dataset cụ thể")
        
        data_choice = input("Chọn (1-2): ").strip()
        
        dataset_names = None
        if data_choice == '2':
            dataset_name = input("Nhập tên dataset (username/dataset-name): ").strip()
            if dataset_name:
                dataset_names = [dataset_name]
        
        success = trainer.run_full_pipeline(dataset_names)
        if success:
            print("🎉 HOÀN THÀNH!")
        else:
            print("❌ THẤT BẠI!")
    
    elif choice == '2':
        # Chỉ tải dữ liệu
        try:
            data_file = trainer.prepare_data_from_kaggle()
            print(f"✅ Dữ liệu đã sẵn sàng: {data_file}")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    elif choice == '3':
        # Training với dữ liệu có sẵn
        training_data_dir = os.path.join(trainer.base_dir, 'training_data')
        json_files = [f for f in os.listdir(training_data_dir) if f.endswith('.json')]
        
        if not json_files:
            print("❌ Không tìm thấy file dữ liệu nào")
        else:
            print(f"\n📋 Tìm thấy {len(json_files)} file dữ liệu:")
            for i, file in enumerate(json_files, 1):
                print(f"  {i}. {file}")
            
            try:
                file_idx = int(input("Chọn file (số): ")) - 1
                if 0 <= file_idx < len(json_files):
                    data_file = os.path.join(training_data_dir, json_files[file_idx])
                    config = trainer.create_han_training_config(data_file)
                    success = trainer.start_han_training(config)
                    print("✅ HOÀN THÀNH!" if success else "❌ THẤT BẠI!")
                else:
                    print("❌ Lựa chọn không hợp lệ")
            except ValueError:
                print("❌ Vui lòng nhập số")
    
    elif choice == '4':
        # Kiểm tra dữ liệu
        training_data_dir = os.path.join(trainer.base_dir, 'training_data')
        json_files = [f for f in os.listdir(training_data_dir) if f.endswith('.json')]
        
        if not json_files:
            print("❌ Không tìm thấy file dữ liệu nào")
        else:
            for file in json_files:
                print(f"\n🔍 Kiểm tra file: {file}")
                data_file = os.path.join(training_data_dir, file)
                trainer.validate_data_for_han(data_file)
    
    else:
        print("❌ Lựa chọn không hợp lệ")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
