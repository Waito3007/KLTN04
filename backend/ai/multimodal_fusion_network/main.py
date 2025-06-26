"""
Điểm vào chính cho pipeline phân tích commit.
Sử dụng script này để chạy pipeline từ command line.
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

from pipeline import CommitAnalysisPipeline, run_end_to_end_pipeline

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline Phân tích Commit')
    
    # Các tham số chung
    parser.add_argument('--base_dir', type=str, default='commit_analysis',
                        help='Thư mục cơ sở cho lưu trữ dữ liệu và mô hình')
    parser.add_argument('--github_token', type=str, default=None,
                        help='Token truy cập GitHub API')
    parser.add_argument('--device', type=str, default=None,
                        help='Thiết bị sử dụng (cuda hoặc cpu)')
    
    # Các subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Subcommand collect
    collect_parser = subparsers.add_parser('collect', help='Thu thập dữ liệu')
    collect_parser.add_argument('--repo_names', type=str, nargs='+', required=True,
                               help='Danh sách tên repository (dạng owner/repo)')
    collect_parser.add_argument('--max_commits', type=int, default=1000,
                               help='Số lượng commit tối đa cho mỗi repo')
    collect_parser.add_argument('--output_file', type=str, default=None,
                               help='Tên file đầu ra')
    
    # Subcommand process
    process_parser = subparsers.add_parser('process', help='Xử lý dữ liệu')
    process_parser.add_argument('--input_file', type=str, required=True,
                               help='Đường dẫn đến file dữ liệu thô')
    process_parser.add_argument('--train_ratio', type=float, default=0.7,
                               help='Tỷ lệ dữ liệu cho tập train')
    process_parser.add_argument('--val_ratio', type=float, default=0.15,
                               help='Tỷ lệ dữ liệu cho tập validation')
    process_parser.add_argument('--test_ratio', type=float, default=0.15,
                               help='Tỷ lệ dữ liệu cho tập test')
    process_parser.add_argument('--no_auto_labeling', action='store_true',
                               help='Không tự động gán nhãn')
    process_parser.add_argument('--random_seed', type=int, default=42,
                               help='Seed ngẫu nhiên')
    
    # Subcommand train
    train_parser = subparsers.add_parser('train', help='Huấn luyện mô hình')
    train_parser.add_argument('--train_path', type=str, required=True,
                             help='Đường dẫn đến file dữ liệu train')
    train_parser.add_argument('--val_path', type=str, required=True,
                             help='Đường dẫn đến file dữ liệu validation')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Kích thước batch')
    train_parser.add_argument('--num_epochs', type=int, default=50,
                             help='Số epochs')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3,
                             help='Learning rate')
    train_parser.add_argument('--early_stopping_patience', type=int, default=5,
                             help='Số epochs chờ trước khi dừng sớm')
    train_parser.add_argument('--text_encoder', type=str, default='transformer', choices=['lstm', 'transformer'],
                             help='Phương thức mã hóa text')
    
    # Subcommand evaluate
    evaluate_parser = subparsers.add_parser('evaluate', help='Đánh giá mô hình')
    evaluate_parser.add_argument('--test_path', type=str, required=True,
                                help='Đường dẫn đến file dữ liệu test')
    evaluate_parser.add_argument('--model_path', type=str, required=True,
                                help='Đường dẫn đến checkpoint của mô hình')
    evaluate_parser.add_argument('--batch_size', type=int, default=32,
                                help='Kích thước batch')
    
    # Subcommand predict
    predict_parser = subparsers.add_parser('predict', help='Dự đoán cho một commit')
    predict_parser.add_argument('--model_path', type=str, required=True,
                               help='Đường dẫn đến checkpoint của mô hình')
    predict_parser.add_argument('--commit_message', type=str, required=True,
                               help='Nội dung commit message')
    predict_parser.add_argument('--metadata_file', type=str, default=None,
                               help='Đường dẫn đến file JSON chứa metadata')
    
    # Subcommand predict_batch
    predict_batch_parser = subparsers.add_parser('predict_batch', help='Dự đoán cho một batch commit')
    predict_batch_parser.add_argument('--model_path', type=str, required=True,
                                     help='Đường dẫn đến checkpoint của mô hình')
    predict_batch_parser.add_argument('--input_file', type=str, required=True,
                                     help='Đường dẫn đến file JSON chứa danh sách commit')
    predict_batch_parser.add_argument('--output_file', type=str, default=None,
                                     help='Đường dẫn đến file đầu ra')
    
    # Subcommand end_to_end
    end_to_end_parser = subparsers.add_parser('end_to_end', help='Chạy toàn bộ pipeline end-to-end')
    end_to_end_parser.add_argument('--repo_names', type=str, nargs='+', required=True,
                                  help='Danh sách tên repository (dạng owner/repo)')
    end_to_end_parser.add_argument('--max_commits', type=int, default=1000,
                                  help='Số lượng commit tối đa cho mỗi repo')
    end_to_end_parser.add_argument('--batch_size', type=int, default=32,
                                  help='Kích thước batch')
    end_to_end_parser.add_argument('--num_epochs', type=int, default=50,
                                  help='Số epochs')
    end_to_end_parser.add_argument('--text_encoder', type=str, default='transformer', choices=['lstm', 'transformer'],
                                  help='Phương thức mã hóa text')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Khởi tạo pipeline
    pipeline = CommitAnalysisPipeline(args.base_dir, args.github_token, args.device)
    
    if args.command == 'collect':
        # Thu thập dữ liệu
        output_file = pipeline.collect_data(
            repo_names=args.repo_names,
            max_commits_per_repo=args.max_commits,
            output_file=args.output_file
        )
        print(f"Đã thu thập dữ liệu và lưu vào: {output_file}")
    
    elif args.command == 'process':
        # Xử lý dữ liệu
        train_path, val_path, test_path = pipeline.process_data(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            auto_labeling=not args.no_auto_labeling,
            random_seed=args.random_seed
        )
        print(f"Đã xử lý dữ liệu và lưu vào:")
        print(f"  Train: {train_path}")
        print(f"  Validation: {val_path}")
        print(f"  Test: {test_path}")
    
    elif args.command == 'train':
        # Huấn luyện mô hình
        model, checkpoint_path = pipeline.train_model(
            train_path=args.train_path,
            val_path=args.val_path,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            text_encoder_method=args.text_encoder
        )
        print(f"Đã huấn luyện mô hình và lưu checkpoint vào: {checkpoint_path}")
    
    elif args.command == 'evaluate':
        # Đánh giá mô hình
        results = pipeline.evaluate_model(
            test_path=args.test_path,
            model_path=args.model_path,
            batch_size=args.batch_size
        )
        print("Kết quả đánh giá:")
        for task_name, task_results in results.items():
            if task_name == 'raw':
                continue
            
            print(f"Task: {task_name}")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
    
    elif args.command == 'predict':
        # Tải predictor
        pipeline.load_predictor(args.model_path)
        
        # Đọc metadata nếu có
        metadata = None
        if args.metadata_file:
            with open(args.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Dự đoán
        result = pipeline.predict(args.commit_message, metadata)
        
        # Hiển thị kết quả
        print("\nDự đoán:")
        for task, task_result in result['prediction'].items():
            if 'label' in task_result:
                print(f"  {task}: {task_result['label']} (Độ tin cậy: {task_result['confidence']:.2f})")
            else:
                print(f"  {task}: {task_result['value']:.2f}")
        
        print("\nĐề xuất:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. [{rec['priority']}] {rec['message']}")
    
    elif args.command == 'predict_batch':
        # Tải predictor
        pipeline.load_predictor(args.model_path)
        
        # Đọc dữ liệu đầu vào
        with open(args.input_file, 'r', encoding='utf-8') as f:
            commit_data = json.load(f)
        
        # Dự đoán batch
        results = pipeline.batch_predict(commit_data)
        
        # Lưu kết quả
        output_file = args.output_file
        if output_file is None:
            output_file = os.path.join(args.base_dir, 'results', 'batch_predictions.json')
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Đã dự đoán cho {len(results)} commits và lưu kết quả vào: {output_file}")
    
    elif args.command == 'end_to_end':
        # Kiểm tra token GitHub
        if args.github_token is None:
            print("Lỗi: Cần cung cấp GitHub token để chạy end-to-end pipeline.")
            return
        
        # Chạy toàn bộ pipeline
        pipeline = run_end_to_end_pipeline(
            base_dir=args.base_dir,
            github_token=args.github_token,
            repo_names=args.repo_names,
            max_commits_per_repo=args.max_commits,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            text_encoder_method=args.text_encoder
        )
        
        print("Đã hoàn thành toàn bộ pipeline end-to-end!")
    
    else:
        print("Vui lòng chọn một lệnh: collect, process, train, evaluate, predict, predict_batch, end_to_end")
        print("Sử dụng --help để xem thêm thông tin.")


if __name__ == "__main__":
    main()
