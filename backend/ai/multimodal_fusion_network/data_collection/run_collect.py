"""
Script đơn giản để khởi chạy quá trình thu thập dữ liệu.
Chạy script này để bắt đầu thu thập commit từ GitHub.
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

def run_collect():
    """Chạy quá trình thu thập commit."""
    
    # Yêu cầu GitHub token
    github_token = input("Nhập GitHub token của bạn: ").strip()
    
    if not github_token:
        logger.error("GitHub token là bắt buộc để thu thập dữ liệu!")
        return
    
    # Tạo thư mục data nếu chưa tồn tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Hỏi số lượng commit cần thu thập
    try:
        target_commits = int(input("Số lượng commit cần thu thập (mặc định 100000): ") or "100000")
    except ValueError:
        logger.warning("Giá trị không hợp lệ, sử dụng mặc định 100000")
        target_commits = 100000
    
    # Hiển thị thông tin
    logger.info(f"Bắt đầu thu thập {target_commits} commit...")
    logger.info(f"Dữ liệu sẽ được lưu vào: {data_dir}")
    logger.info("Quá trình này có thể mất nhiều thời gian (8-24 giờ tùy thuộc vào số lượng commit)")
    logger.info("Bạn có thể ngắt quá trình bất kỳ lúc nào bằng Ctrl+C, và tiếp tục sau")
    
    # Xác nhận
    confirm = input("\nBạn có chắc chắn muốn bắt đầu thu thập? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("Đã hủy thu thập dữ liệu")
        return
    
    # Chạy lệnh thu thập
    collect_script = os.path.join(current_dir, "collect_100k.py")
    
    try:
        command = [
            sys.executable,
            collect_script,
            "collect",
            "--token", github_token,
            "--output_dir", data_dir,
            "--target", str(target_commits)
        ]
        
        logger.info("Đang khởi chạy quá trình thu thập...")
        subprocess.run(command)
        
    except KeyboardInterrupt:
        logger.info("\nQuá trình thu thập đã bị ngắt bởi người dùng")
        logger.info("Bạn có thể tiếp tục sau bằng cách chạy lại script này")
    except Exception as e:
        logger.error(f"Lỗi khi chạy quá trình thu thập: {str(e)}")

if __name__ == "__main__":
    run_collect()
