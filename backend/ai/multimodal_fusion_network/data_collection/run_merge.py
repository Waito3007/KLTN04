"""
Script đơn giản để kết hợp các file commit đã thu thập thành một file duy nhất.
Chạy script này sau khi đã thu thập dữ liệu bằng run_collect.py.
"""
import os
import logging
import subprocess
import sys

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_merge():
    """Chạy quá trình kết hợp các file commit."""
    
    # Xác định thư mục dữ liệu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    if not os.path.exists(data_dir):
        logger.error(f"Thư mục dữ liệu {data_dir} không tồn tại!")
        return
    
    # Đếm số file commit
    commit_files = [f for f in os.listdir(data_dir) if f.startswith("github_commits_") and f.endswith(".json")]
    
    if not commit_files:
        logger.error(f"Không tìm thấy file commit nào trong {data_dir}")
        logger.error("Hãy chạy run_collect.py trước!")
        return
    
    logger.info(f"Tìm thấy {len(commit_files)} file commit trong {data_dir}")
    
    # Xác định file đầu ra
    output_file = os.path.join(data_dir, "all_commits_merged.json")
    logger.info(f"File đầu ra: {output_file}")
    
    # Xác nhận
    confirm = input(f"\nBạn có chắc chắn muốn kết hợp {len(commit_files)} file thành một? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("Đã hủy quá trình kết hợp")
        return
    
    # Chạy lệnh kết hợp
    collect_script = os.path.join(current_dir, "collect_100k.py")
    
    try:
        command = [
            sys.executable,
            collect_script,
            "merge",
            "--input_dir", data_dir,
            "--output_file", output_file
        ]
        
        logger.info("Đang kết hợp các file commit...")
        subprocess.run(command)
        
        if os.path.exists(output_file):
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            logger.info(f"Đã kết hợp thành công! Kích thước file: {file_size_mb:.2f} MB")
        
    except KeyboardInterrupt:
        logger.info("\nQuá trình kết hợp đã bị ngắt bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi khi chạy quá trình kết hợp: {str(e)}")

if __name__ == "__main__":
    run_merge()
