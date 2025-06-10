#!/usr/bin/env python3
"""
Script setup để cài đặt dependencies và cấu hình Kaggle API
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Cài đặt package Python"""
    print(f"📦 Đang cài đặt {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Đã cài đặt {package} thành công")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Lỗi khi cài đặt {package}")
        return False

def check_kaggle_setup():
    """Kiểm tra và hướng dẫn setup Kaggle API"""
    print("\n🔧 KIỂM TRA KAGGLE API SETUP")
    print("=" * 50)
    
    # Kiểm tra kaggle package
    try:
        import kaggle
        print("✅ Kaggle package đã được cài đặt")
    except ImportError:
        print("❌ Kaggle package chưa được cài đặt")
        if install_package("kaggle"):
            import kaggle
        else:
            return False
    
    # Kiểm tra kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_config = kaggle_dir / "kaggle.json"
    
    if kaggle_config.exists():
        print("✅ File kaggle.json đã tồn tại")
        
        # Kiểm tra quyền file (trên Linux/Mac)
        if os.name != 'nt':  # Không phải Windows
            stat_info = kaggle_config.stat()
            if stat_info.st_mode & 0o077:
                print("⚠️ File kaggle.json có quyền không an toàn")
                print("🔧 Đang sửa quyền file...")
                kaggle_config.chmod(0o600)
                print("✅ Đã sửa quyền file kaggle.json")
        
        # Test API
        try:
            kaggle.api.authenticate()
            print("✅ Kaggle API hoạt động bình thường")
            return True
        except Exception as e:
            print(f"❌ Lỗi xác thực Kaggle API: {e}")
            return False
    else:
        print("❌ File kaggle.json không tồn tại")
        print("\n📋 HƯỚNG DẪN SETUP KAGGLE API:")
        print("1. Truy cập: https://www.kaggle.com/settings")
        print("2. Scroll xuống phần 'API' và click 'Create New API Token'")
        print("3. Download file kaggle.json")
        print("4. Đặt file vào thư mục:")
        print(f"   • Windows: {Path.home() / '.kaggle'}")
        print(f"   • Linux/Mac: ~/.kaggle/")
        print("5. Chạy lại script này")
        
        # Tạo thư mục .kaggle nếu chưa có
        kaggle_dir.mkdir(exist_ok=True)
        print(f"\n📁 Đã tạo thư mục: {kaggle_dir}")
        
        return False

def install_required_packages():
    """Cài đặt các package cần thiết"""
    print("🚀 CÀI ĐẶT CÁC PACKAGE CẦN THIẾT")
    print("=" * 50)
    
    required_packages = [
        "kaggle",
        "pandas",
        "numpy",
        "requests",
        "tqdm"
    ]
    
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Kết quả: {success_count}/{len(required_packages)} packages được cài đặt thành công")
    return success_count == len(required_packages)

def test_dataset_download():
    """Test tải một dataset nhỏ"""
    print("\n🧪 TEST TẢI DATASET MẪU")
    print("=" * 50)
    
    try:
        import kaggle
        
        # Tạo thư mục test
        test_dir = Path("test_download")
        test_dir.mkdir(exist_ok=True)
        
        print("🔄 Đang test tải dataset mẫu...")
        
        # Tải một dataset nhỏ để test
        kaggle.api.dataset_download_files(
            "shashankbansal6/git-commits-message-dataset",
            path=str(test_dir),
            unzip=True
        )
        
        print("✅ Test tải dataset thành công!")
        
        # Dọn dẹp
        import shutil
        shutil.rmtree(test_dir)
        print("🗑️ Đã dọn dẹp file test")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi test tải dataset: {e}")
        return False

def main():
    """Hàm chính"""
    print("🔧 SETUP KAGGLE DATASET DOWNLOADER")
    print("=" * 80)
    
    # Bước 1: Cài đặt packages
    packages_ok = install_required_packages()
    
    if not packages_ok:
        print("❌ Không thể cài đặt tất cả packages cần thiết")
        return
    
    # Bước 2: Setup Kaggle API
    kaggle_ok = check_kaggle_setup()
    
    if not kaggle_ok:
        print("❌ Kaggle API chưa được setup đúng cách")
        print("👆 Vui lòng làm theo hướng dẫn ở trên")
        return
    
    # Bước 3: Test tải dataset
    test_ok = test_dataset_download()
    
    if test_ok:
        print("\n🎉 SETUP HOÀN TẤT!")
        print("✅ Bạn có thể chạy script download_kaggle_dataset.py")
    else:
        print("\n⚠️ Setup cơ bản hoàn tất nhưng có lỗi khi test")
        print("💡 Hãy thử chạy script download_kaggle_dataset.py để xem chi tiết")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
