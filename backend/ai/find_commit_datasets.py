"""
Tìm kiếm và download các dataset commit messages có sẵn trên Kaggle
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def setup_kaggle_api():
    """Setup Kaggle API"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"❌ Lỗi setup Kaggle API: {e}")
        return None

def search_commit_datasets(api):
    """Tìm kiếm datasets về commit messages"""
    try:
        print("🔍 TÌNG KIẾM DATASETS VỀ COMMIT MESSAGES")
        print("="*60)
        
        # Tìm kiếm với từ khóa khác nhau
        search_terms = ['commit', 'git commit', 'github commit', 'commit message', 'git']
        all_datasets = []
        
        for term in search_terms:
            print(f"\n🔎 Tìm kiếm: '{term}'")
            try:
                datasets = api.dataset_list(search=term, page_size=10)
                for dataset in datasets:
                    # Lọc những dataset có vẻ liên quan đến commit
                    title_lower = dataset.title.lower()
                    ref_lower = dataset.ref.lower()
                    
                    if any(keyword in title_lower or keyword in ref_lower 
                           for keyword in ['commit', 'git', 'github', 'message']):
                        
                        dataset_info = {
                            'ref': dataset.ref,
                            'title': dataset.title,
                            'size': dataset.totalBytes,
                            'downloads': dataset.downloadCount,
                            'files': dataset.fileCount,
                            'license': dataset.licenseName,
                            'updated': dataset.lastUpdated
                        }
                        
                        # Kiểm tra duplicate
                        if not any(d['ref'] == dataset_info['ref'] for d in all_datasets):
                            all_datasets.append(dataset_info)
                            
            except Exception as e:
                print(f"  ⚠️ Lỗi tìm kiếm '{term}': {e}")
        
        # Sắp xếp theo downloads
        all_datasets.sort(key=lambda x: x['downloads'], reverse=True)
        
        print(f"\n📋 TÌNG THẤY {len(all_datasets)} DATASETS:")
        print("="*80)
        
        for i, dataset in enumerate(all_datasets[:15], 1):  # Top 15
            size_mb = dataset['size'] / (1024*1024) if dataset['size'] else 0
            print(f"{i:2d}. {dataset['ref']}")
            print(f"    📄 {dataset['title']}")
            print(f"    📊 {dataset['downloads']:,} downloads | {size_mb:.1f} MB | {dataset['files']} files")
            print(f"    📅 Updated: {dataset['updated']}")
            print()
        
        return all_datasets
        
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm datasets: {e}")
        return []

def test_dataset_access(api, dataset_ref):
    """Test xem có thể download dataset không"""
    try:
        print(f"🔍 Testing access: {dataset_ref}")
        
        # Thử lấy thông tin files
        files = api.dataset_list_files(dataset_ref)
        print(f"  ✅ Files accessible: {len(files)}")
        
        for file in files[:3]:  # Show first 3 files
            size_mb = file.totalBytes / (1024*1024) if file.totalBytes else 0
            print(f"    📄 {file.name} ({size_mb:.1f} MB)")
        
        return True, files
        
    except Exception as e:
        print(f"  ❌ Access denied: {e}")
        return False, []

def download_and_preview_dataset(api, dataset_ref, max_preview_rows=1000):
    """Download và preview dataset"""
    try:
        print(f"\n📥 DOWNLOADING: {dataset_ref}")
        print("="*60)
        
        # Tạo thư mục download
        download_dir = Path(__file__).parent / "kaggle_data" / dataset_ref.replace('/', '_')
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download
        print(f"📁 Download to: {download_dir}")
        api.dataset_download_files(dataset_ref, path=str(download_dir), unzip=True)
        
        # Tìm files CSV/JSON
        csv_files = list(download_dir.glob("*.csv"))
        json_files = list(download_dir.glob("*.json"))
        txt_files = list(download_dir.glob("*.txt"))
        
        print(f"📋 Files downloaded:")
        all_files = csv_files + json_files + txt_files
        for file in all_files:
            size_mb = file.stat().st_size / (1024*1024)
            print(f"  📄 {file.name} ({size_mb:.1f} MB)")
        
        # Preview CSV files
        for csv_file in csv_files[:2]:  # Preview first 2 CSV files
            try:
                print(f"\n📊 PREVIEW: {csv_file.name}")
                print("-"*50)
                
                # Read sample
                df = pd.read_csv(csv_file, nrows=max_preview_rows)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                # Look for commit-related columns
                commit_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in 
                           ['message', 'commit', 'subject', 'title', 'description']):
                        commit_cols.append(col)
                
                if commit_cols:
                    print(f"🎯 Potential commit columns: {commit_cols}")
                    
                    # Show samples
                    for col in commit_cols[:2]:
                        print(f"\nSample {col}:")
                        samples = df[col].dropna().head(5)
                        for i, sample in enumerate(samples, 1):
                            sample_str = str(sample)[:100]
                            print(f"  {i}. {sample_str}{'...' if len(str(sample)) > 100 else ''}")
                
                print(f"✅ {csv_file.name} looks promising for commit data")
                
            except Exception as e:
                print(f"⚠️ Error previewing {csv_file.name}: {e}")
        
        return download_dir, all_files
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None, []

def main():
    """Main function"""
    print("🔍 KAGGLE COMMIT DATASETS FINDER")
    print("="*60)
    
    # Setup API
    api = setup_kaggle_api()
    if not api:
        return
    
    # Search datasets
    datasets = search_commit_datasets(api)
    if not datasets:
        print("❌ Không tìm thấy datasets phù hợp")
        return
    
    # Let user choose
    print("\n🎯 CHỌN DATASET ĐỂ DOWNLOAD:")
    print("0. Exit")
    for i, dataset in enumerate(datasets[:10], 1):
        print(f"{i}. {dataset['ref']} ({dataset['downloads']:,} downloads)")
    
    choice = input("\nNhập số dataset muốn download (0-10): ").strip()
    
    if choice == '0' or not choice.isdigit():
        print("👋 Goodbye!")
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < min(10, len(datasets)):
            selected = datasets[idx]
            dataset_ref = selected['ref']
            
            print(f"\n✨ ĐÃ CHỌN: {dataset_ref}")
            
            # Test access
            accessible, files = test_dataset_access(api, dataset_ref)
            if not accessible:
                print("❌ Dataset không thể truy cập")
                return
            
            # Download and preview
            download_dir, downloaded_files = download_and_preview_dataset(api, dataset_ref)
            
            if download_dir and downloaded_files:
                print(f"\n🎉 THÀNH CÔNG!")
                print(f"📁 Data downloaded to: {download_dir}")
                print(f"📋 Bây giờ bạn có thể:")
                print(f"  • Kiểm tra files trong {download_dir}")
                print(f"  • Sử dụng data để train model")
                
                # Suggestion for next steps
                csv_files = [f for f in downloaded_files if f.suffix == '.csv']
                if csv_files:
                    print(f"\n💡 ĐỀ XUẤT:")
                    print(f"  Sử dụng file CSV chính: {csv_files[0].name}")
                    print(f"  Có thể train model với data này")
            
        else:
            print("❌ Lựa chọn không hợp lệ")
            
    except ValueError:
        print("❌ Vui lòng nhập số hợp lệ")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
