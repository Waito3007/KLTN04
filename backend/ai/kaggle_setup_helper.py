"""
Kaggle API Setup Helper
Hướng dẫn và hỗ trợ setup Kaggle API một cách tự động
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def create_kaggle_directory():
    """Tạo thư mục .kaggle nếu chưa có"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    print(f"✅ Đã tạo thư mục: {kaggle_dir}")
    return kaggle_dir

def check_kaggle_json_exists():
    """Kiểm tra xem kaggle.json đã tồn tại chưa"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    return kaggle_json.exists(), kaggle_json

def install_kaggle_package():
    """Cài đặt kaggle package"""
    try:
        print("📦 Đang cài đặt kaggle package...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], 
                      check=True, capture_output=True, text=True)
        print("✅ Đã cài đặt kaggle package thành công")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi cài đặt kaggle: {e}")
        return False

def create_sample_kaggle_json():
    """Tạo file kaggle.json mẫu"""
    kaggle_dir = create_kaggle_directory()
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    sample_config = {
        "username": "YOUR_KAGGLE_USERNAME",
        "key": "YOUR_KAGGLE_API_KEY"
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    # Set permissions (Windows thì không cần chmod)
    if os.name != 'nt':
        os.chmod(kaggle_json, 0o600)
    
    print(f"📝 Đã tạo file mẫu: {kaggle_json}")
    return kaggle_json

def get_kaggle_credentials_interactive():
    """Lấy thông tin đăng nhập Kaggle từ user"""
    print("\n" + "="*60)
    print("🔑 NHẬP THÔNG TIN KAGGLE API")
    print("="*60)
    print("Để lấy API credentials, làm theo các bước sau:")
    print("1. Truy cập: https://www.kaggle.com/settings")
    print("2. Scroll xuống phần 'API'")
    print("3. Click 'Create New API Token'")
    print("4. Download file kaggle.json")
    print("5. Mở file và copy thông tin vào đây")
    print("-"*60)
    
    username = input("Nhập Kaggle username: ").strip()
    api_key = input("Nhập Kaggle API key: ").strip()
    
    if not username or not api_key:
        print("❌ Username và API key không được để trống!")
        return None, None
    
    return username, api_key

def save_kaggle_credentials(username, api_key):
    """Lưu thông tin đăng nhập vào kaggle.json"""
    kaggle_dir = create_kaggle_directory()
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    config = {
        "username": username,
        "key": api_key
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set permissions
    if os.name != 'nt':
        os.chmod(kaggle_json, 0o600)
    
    print(f"✅ Đã lưu thông tin vào: {kaggle_json}")
    return kaggle_json

def test_kaggle_connection():
    """Test kết nối Kaggle API"""
    try:
        print("\n🔍 Đang test kết nối Kaggle API...")
        
        # Import sau khi đã setup
        import kaggle.api
        kaggle.api.authenticate()
        
        # Test bằng cách list datasets
        print("📊 Đang lấy danh sách datasets...")
        datasets = kaggle.api.dataset_list(search='commit', page_size=5)
        
        print("✅ Kết nối Kaggle API thành công!")
        print("\n📋 Top 5 commit datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset.ref} ({dataset.downloadCount} downloads)")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi kết nối Kaggle API: {str(e)}")
        return False

def setup_environment_variables():
    """Setup biến môi trường cho Kaggle (alternative method)"""
    print("\n" + "="*60)
    print("🌍 SETUP BIẾN MÔI TRƯỜNG (PHƯƠNG PHÁP THAY THẾ)")
    print("="*60)
    print("Nếu không muốn dùng file kaggle.json, bạn có thể set biến môi trường:")
    
    username = input("Nhập Kaggle username (hoặc Enter để bỏ qua): ").strip()
    if username:
        api_key = input("Nhập Kaggle API key: ").strip()
        if api_key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = api_key
            print("✅ Đã set biến môi trường cho session hiện tại")
            print("⚠️  Lưu ý: Biến môi trường chỉ có hiệu lực trong session này")
            print("   Để permanent, thêm vào System Environment Variables")
            return True
    
    print("ℹ️  Bỏ qua setup biến môi trường")
    return False

def main():
    """Main function"""
    print("🚀 KAGGLE API SETUP HELPER")
    print("="*60)
    
    # Cài đặt kaggle package nếu chưa có
    try:
        import kaggle
        print("✅ Kaggle package đã được cài đặt")
    except ImportError:
        print("📦 Kaggle package chưa được cài đặt")
        if not install_kaggle_package():
            print("❌ Không thể cài đặt kaggle package. Vui lòng cài thủ công:")
            print("   pip install kaggle")
            return
    
    # Kiểm tra kaggle.json
    exists, kaggle_json_path = check_kaggle_json_exists()
    
    if exists:
        print(f"✅ File kaggle.json đã tồn tại: {kaggle_json_path}")
        
        # Test kết nối
        if test_kaggle_connection():
            print("\n🎉 Setup hoàn tất! Kaggle API đã sẵn sàng sử dụng.")
            return
        else:
            print("\n⚠️  File kaggle.json tồn tại nhưng có vấn đề với credentials")
            choice = input("Bạn có muốn setup lại không? (y/n): ").lower()
            if choice != 'y':
                return
    
    print(f"\n📁 Thư mục Kaggle: {Path.home() / '.kaggle'}")
    print("📄 File cần tạo: kaggle.json")
    
    # Hỏi phương pháp setup
    print("\n🔧 Chọn phương pháp setup:")
    print("1. Nhập thông tin API trực tiếp (Khuyên dùng)")
    print("2. Tạo file mẫu để bạn tự điền")
    print("3. Setup biến môi trường")
    print("4. Hướng dẫn manual setup")
    
    choice = input("\nNhập lựa chọn (1-4): ").strip()
    
    if choice == '1':
        username, api_key = get_kaggle_credentials_interactive()
        if username and api_key:
            save_kaggle_credentials(username, api_key)
            if test_kaggle_connection():
                print("\n🎉 Setup thành công!")
            else:
                print("\n❌ Có lỗi với credentials. Vui lòng kiểm tra lại.")
    
    elif choice == '2':
        kaggle_json = create_sample_kaggle_json()
        print(f"\n📝 Đã tạo file mẫu: {kaggle_json}")
        print("🔧 Vui lòng:")
        print("1. Mở file này")
        print("2. Thay thế YOUR_KAGGLE_USERNAME và YOUR_KAGGLE_API_KEY")
        print("3. Lưu file")
        print("4. Chạy lại script này để test")
    
    elif choice == '3':
        if setup_environment_variables():
            if test_kaggle_connection():
                print("\n🎉 Setup thành công!")
    
    elif choice == '4':
        print("\n📋 HƯỚNG DẪN MANUAL SETUP:")
        print("="*40)
        print("1. Truy cập: https://www.kaggle.com/settings")
        print("2. Scroll xuống phần 'API'")
        print("3. Click 'Create New API Token'")
        print("4. Download file kaggle.json")
        print(f"5. Đặt file vào: {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("6. Chạy lại script này để test")
    
    else:
        print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main()
