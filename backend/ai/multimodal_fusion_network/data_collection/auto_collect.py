"""
Script tự động thu thập dữ liệu commit GitHub với khả năng đợi hết rate limit và tiếp tục.
Sử dụng script này thay cho collect_100k.py để tự động tiếp tục sau khi hết rate limit.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_collect.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def auto_collect_commits(github_token: str = None, token_file: str = None, output_dir: str = "data/github_commits", 
                         max_retries: int = 5, 
                         check_interval: int = 60,
                         target_commits: int = 100000):
    """
    Tự động thu thập commit từ GitHub với khả năng tự động đợi và tiếp tục sau khi rate limit reset.
    
    Args:
        github_token: GitHub API token hoặc danh sách token
        token_file: Đường dẫn đến file chứa danh sách token
        output_dir: Thư mục để lưu dữ liệu
        max_retries: Số lần thử lại tối đa khi gặp lỗi
        check_interval: Thời gian (giây) giữa các lần kiểm tra khi đợi rate limit reset
        target_commits: Số lượng commit mục tiêu cần thu thập
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Thêm thư mục gốc vào path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    
    from data_collection.collect_100k import collect_100k_commits
    
    # Đường dẫn đến file trạng thái
    state_file = os.path.join(output_dir, "collection_state.json")
    
    # Đếm số lần thử lại
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Chạy thu thập commit
            logger.info(f"Bắt đầu thu thập commit (lần thử {retry_count + 1}/{max_retries})")
            collect_100k_commits(github_token=github_token, token_file=token_file, output_dir=output_dir, target_commits=target_commits)
            
            # Kiểm tra xem đã hoàn thành chưa hay bị gián đoạn do rate limit
            if os.path.exists(state_file):
                try:
                    with safe_open(state_file, 'r') as f:
                        state = json.load(f)
                
                    if state.get("last_error") == "rate_limit_exceeded":
                        # Tính thời gian cần đợi
                        reset_time_str = state.get("reset_time")
                        if reset_time_str:
                            try:
                                reset_time = datetime.strptime(reset_time_str, '%Y-%m-%d %H:%M:%S')
                                now = datetime.now()
                                wait_seconds = max((reset_time - now).total_seconds(), 0)
                                
                                if wait_seconds > 0:
                                    wait_time_formatted = time.strftime('%H:%M:%S', time.gmtime(wait_seconds))
                                    logger.info(f"Đợi rate limit reset: {wait_time_formatted} (HH:MM:SS)")
                                    logger.info(f"Sẽ tiếp tục vào: {reset_time_str}")
                                    
                                    # Đợi đến thời điểm reset với kiểm tra định kỳ
                                    while datetime.now() < reset_time:
                                        remaining = max((reset_time - datetime.now()).total_seconds(), 0)
                                        remaining_formatted = time.strftime('%H:%M:%S', time.gmtime(remaining))
                                        logger.info(f"Còn lại: {remaining_formatted}")
                                        
                                        # Đợi một khoảng thời gian
                                        time.sleep(min(check_interval, remaining + 1))
                                    
                                    # Thêm 10 giây để đảm bảo rate limit đã được reset hoàn toàn
                                    logger.info("Đợi thêm 10 giây để đảm bảo rate limit đã được reset hoàn toàn")
                                    time.sleep(10)
                                    
                                    # Tiếp tục thu thập (sẽ bắt đầu lại vòng lặp)
                                    continue
                            except (ValueError, TypeError):
                                logger.warning(f"Không thể phân tích thời gian reset: {reset_time_str}")
                                # Đợi 1 giờ nếu không phân tích được thời gian reset
                                logger.info("Đợi 1 giờ trước khi thử lại")
                                time.sleep(3600)
                                continue
                        else:
                            # Nếu không có thời gian reset, đợi 1 giờ
                            logger.info("Không có thông tin về thời gian reset. Đợi 1 giờ trước khi thử lại")
                            time.sleep(3600)
                            continue
                    else:
                        # Nếu không phải lỗi rate limit, kiểm tra xem đã hoàn thành chưa
                        logger.info("Thu thập đã hoàn thành hoặc bị gián đoạn vì lý do khác")
                        break
                except Exception as e:
                    logger.error(f"Lỗi khi đọc state file: {str(e)}")
                    # Nếu không đọc được state file, đợi 10 phút rồi thử lại
                    logger.info("Đợi 10 phút trước khi thử lại")
                    time.sleep(600)
                    continue
            else:
                # Nếu không có file trạng thái, giả định là hoàn thành
                logger.info("Không tìm thấy file trạng thái. Giả định thu thập đã hoàn thành")
                break
                
        except KeyboardInterrupt:
            logger.info("Thu thập bị ngắt bởi người dùng")
            break
        except Exception as e:
            logger.error(f"Lỗi không mong đợi: {str(e)}")
            retry_count += 1
            
            if retry_count < max_retries:
                wait_time = 60 * retry_count  # Tăng thời gian đợi theo số lần thử
                logger.info(f"Đợi {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                logger.error(f"Đã thử {max_retries} lần không thành công. Dừng thu thập.")
                break
    
    logger.info("Kết thúc quy trình thu thập")

def safe_open(file_path, mode='r', encoding='utf-8', **kwargs):
    """
    Mở file một cách an toàn với xử lý lỗi encoding.
    
    Args:
        file_path: Đường dẫn đến file
        mode: Chế độ mở file ('r', 'w', 'a', etc.)
        encoding: Encoding sử dụng (mặc định là utf-8)
        **kwargs: Các tham số khác cho hàm open
        
    Returns:
        File object
    """
    try:
        # Sử dụng đường dẫn tuyệt đối
        abs_path = os.path.abspath(file_path)
        return open(abs_path, mode=mode, encoding=encoding, **kwargs)
    except UnicodeEncodeError as e:
        # Xử lý lỗi Unicode trong đường dẫn
        logger.error(f"Lỗi Unicode khi mở file: {str(e)}")
        logger.info("Thử chuyển đổi đường dẫn sang dạng ngắn gọn...")
        
        try:
            # Thử sử dụng short path name trong Windows
            if os.name == 'nt':  # Windows
                import ctypes
                buffer_size = 500
                buffer = ctypes.create_unicode_buffer(buffer_size)
                get_short_path_name = ctypes.windll.kernel32.GetShortPathNameW
                get_short_path_name(abs_path, buffer, buffer_size)
                short_path = buffer.value
                logger.info(f"Đã chuyển đổi sang đường dẫn ngắn: {short_path}")
                return open(short_path, mode=mode, encoding=encoding, **kwargs)
            else:
                raise e  # Nếu không phải Windows, ném lại ngoại lệ
        except Exception as e2:
            logger.error(f"Không thể mở file với đường dẫn ngắn: {str(e2)}")
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tự động thu thập commit từ GitHub với xử lý rate limit')
    parser.add_argument('--token', help='GitHub API token')
    parser.add_argument('--token_file', help='Đường dẫn đến file chứa danh sách token')
    parser.add_argument('--output_dir', default='data/github_commits', help='Thư mục lưu dữ liệu')
    parser.add_argument('--max_retries', type=int, default=5, help='Số lần thử lại tối đa')
    parser.add_argument('--check_interval', type=int, default=60, help='Thời gian (giây) giữa các lần kiểm tra rate limit')
    parser.add_argument('--target', type=int, default=100000, help='Số lượng commit mục tiêu')
    
    args = parser.parse_args()
    
    # Ưu tiên token file nếu cả hai được cung cấp
    github_token = args.token
    token_file = args.token_file
    
    # Nếu không có token và token_file
    if not github_token and not token_file:
        # Thử tìm file token mặc định
        default_token_file = os.path.join("data", "tokens.txt")
        if os.path.exists(default_token_file):
            token_file = default_token_file
            logger.info(f"Sử dụng file token mặc định: {default_token_file}")
        else:
            # Kiểm tra biến môi trường
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                logger.warning("Không tìm thấy token nào. Vui lòng cung cấp token qua tham số, file token hoặc biến môi trường GITHUB_TOKEN")
                logger.info("Ví dụ: python auto_collect.py --token ghp_your_token")
                logger.info("Ví dụ: python auto_collect.py --token_file data/tokens.txt")
                sys.exit(1)
    
    # Bắt đầu thu thập tự động
    auto_collect_commits(
        github_token=github_token, 
        token_file=token_file,
        output_dir=args.output_dir, 
        max_retries=args.max_retries,
        check_interval=args.check_interval,
        target_commits=args.target
    )
