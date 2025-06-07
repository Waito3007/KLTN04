"""
Quick Kaggle Setup - Chỉ tạo file kaggle.json
"""

import json
from pathlib import Path

def main():
    print("🚀 QUICK KAGGLE SETUP")
    print("="*50)
    
    # Tạo thư mục .kaggle
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    print(f"✅ Thư mục: {kaggle_dir}")
    
    # Kiểm tra file đã tồn tại chưa
    kaggle_json = kaggle_dir / 'kaggle.json'
    if kaggle_json.exists():
        print(f"✅ File kaggle.json đã tồn tại")
        
        # Đọc và hiển thị thông tin hiện tại
        try:
            with open(kaggle_json, 'r') as f:
                config = json.load(f)
            print(f"📋 Username hiện tại: {config.get('username', 'N/A')}")
            
            choice = input("Bạn có muốn cập nhật lại không? (y/n): ").lower()
            if choice != 'y':
                print("✅ Giữ nguyên config hiện tại")
                return
        except Exception as e:
            print(f"⚠️  File có lỗi: {e}")
    
    # Hướng dẫn lấy credentials
    print("\n📋 HƯỚNG DẪN LẤY KAGGLE API:")
    print("1. Truy cập: https://www.kaggle.com/settings")
    print("2. Scroll xuống phần 'API'")
    print("3. Click 'Create New API Token'")
    print("4. Download file kaggle.json")
    print("5. Mở file và copy thông tin vào đây")
    print("-" * 50)
    
    # Nhập thông tin
    username = input("Nhập username: ").strip()
    if not username:
        print("❌ Username không được để trống!")
        return
    
    api_key = input("Nhập API key: ").strip()
    if not api_key:
        print("❌ API key không được để trống!")
        return
    
    # Tạo config
    config = {
        "username": username,
        "key": api_key
    }
    
    # Lưu file
    try:
        with open(kaggle_json, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Đã lưu: {kaggle_json}")
        
        # Test đơn giản
        print("\n🔍 Test import kaggle...")
        try:
            import kaggle
            print("✅ Import kaggle thành công!")
            print("\n🎉 Setup hoàn tất!")
            print("Bây giờ bạn có thể chạy các script khác")
        except Exception as e:
            print(f"⚠️  Import kaggle lỗi: {e}")
            print("Nhưng file kaggle.json đã được tạo")
            
    except Exception as e:
        print(f"❌ Lỗi lưu file: {e}")

if __name__ == "__main__":
    main()
