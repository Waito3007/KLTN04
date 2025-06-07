"""
Simple Kaggle API Setup Script
Thiết lập Kaggle API một cách đơn giản
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

def get_kaggle_credentials():
    """Lấy thông tin Kaggle API từ user"""
    print("\n" + "="*60)
    print("🔑 SETUP KAGGLE API CREDENTIALS")
    print("="*60)
    print("📋 Hướng dẫn lấy API credentials:")
    print("1. Truy cập: https://www.kaggle.com/settings")
    print("2. Scroll xuống phần 'API'")
    print("3. Click 'Create New API Token'")
    print("4. Download file kaggle.json")
    print("5. Mở file và copy username + key vào đây")
    print("-"*60)
    
    username = input("Nhập Kaggle username: ").strip()
    if not username:
        print("❌ Username không được để trống!")
        return None, None
        
    api_key = input("Nhập Kaggle API key: ").strip()
    if not api_key:
        print("❌ API key không được để trống!")
        return None, None
    
    return username, api_key

def save_kaggle_json(username, api_key):
    """Lưu thông tin vào kaggle.json"""
    kaggle_dir = create_kaggle_directory()
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    config = {
        "username": username,
        "key": api_key
    }
    
    try:
        with open(kaggle_json, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Đã lưu thông tin vào: {kaggle_json}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
        return False

def test_kaggle_api():
    """Test Kaggle API"""
    try:
        print("\n🔍 Đang test Kaggle API...")
        
        # Import kaggle sau khi đã setup file
        os.environ['KAGGLE_CONFIG_DIR'] = str(Path.home() / '.kaggle')
        
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Test với dataset search
        datasets = list(api.dataset_list(search='commit', page_size=3))
        
        print("✅ Kaggle API hoạt động thành công!")
        print("\n📋 Một số dataset commit có sẵn:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset.ref}")
            print(f"     Downloads: {dataset.downloadCount}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test API: {str(e)}")
        return False

def install_dependencies():
    """Cài đặt dependencies cần thiết"""
    packages = ['kaggle', 'pandas', 'numpy']
    
    print("📦 Đang cài đặt dependencies...")
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} đã có sẵn")
        except ImportError:
            try:
                print(f"📥 Đang cài đặt {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"✅ Đã cài {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Lỗi cài {package}: {e}")
                return False
    return True

def main():
    """Main function"""
    print("🚀 SIMPLE KAGGLE SETUP")
    print("="*50)
    
    # Kiểm tra xem kaggle.json đã tồn tại chưa
    kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json_path.exists():
        print(f"✅ File kaggle.json đã tồn tại: {kaggle_json_path}")
        
        # Test API
        if test_kaggle_api():
            print("\n🎉 Kaggle API đã sẵn sàng sử dụng!")
            return
        else:
            print("\n⚠️  File tồn tại nhưng có vấn đề. Bạn có muốn setup lại không?")
            choice = input("Setup lại? (y/n): ").lower()
            if choice != 'y':
                return
    
    # Cài đặt dependencies
    if not install_dependencies():
        print("❌ Không thể cài đặt dependencies")
        return
    
    # Lấy credentials
    username, api_key = get_kaggle_credentials()
    if not username or not api_key:
        print("❌ Thiếu thông tin credentials")
        return
    
    # Lưu file
    if not save_kaggle_json(username, api_key):
        print("❌ Không thể lưu file kaggle.json")
        return
    
    # Test API
    if test_kaggle_api():
        print("\n🎉 Setup thành công! Kaggle API đã sẵn sàng!")
        print("\n📋 Bây giờ bạn có thể:")
        print("  • Chạy: python download_kaggle_dataset.py")
        print("  • Hoặc: python train_han_with_kaggle.py")
    else:
        print("❌ Setup không thành công. Vui lòng kiểm tra lại credentials.")

if __name__ == "__main__":
    main()
