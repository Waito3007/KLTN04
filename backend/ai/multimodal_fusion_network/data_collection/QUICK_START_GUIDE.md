# Hướng dẫn thu thập commit GitHub với xử lý rate limit tự động

## Chuẩn bị

1. Đảm bảo bạn đã cài đặt thư viện cần thiết:

   ```
   pip install requests
   ```

2. Chuẩn bị GitHub token:
   - Tạo token tại: https://github.com/settings/tokens
   - Cần cấp quyền: `public_repo`
   - Lưu token vào file `data/tokens.txt` hoặc sử dụng tham số dòng lệnh

## Thu thập tự động với xử lý rate limit

### Bước 1: Chuẩn bị token (chọn một trong các cách)

**Cách 1: Sử dụng file tokens.txt**

```
# Tạo thư mục data nếu chưa có
mkdir -p data

# Tạo file tokens.txt với một token mỗi dòng
echo "github_pat_your_token_here" > data/tokens.txt
```

**Cách 2: Đặt biến môi trường**

```powershell
# Windows PowerShell
$env:GITHUB_TOKEN="ghp_your_token_here"

# Windows CMD
set GITHUB_TOKEN=ghp_your_token_here
```

**Cách 3: Sử dụng tham số dòng lệnh**

```
python auto_collect.py --token ghp_your_token_here
```

### Bước 2: Chạy script thu thập tự động

```
cd backend/ai/multimodal_fusion_network/data_collection
python auto_collect.py
```

Hoặc với tùy chọn:

```
python auto_collect.py --token_file ../../data/tokens.txt --output_dir ../../data/github_commits --target 100000
```

### Quá trình hoạt động

1. Script sẽ bắt đầu thu thập commit từ các repository phổ biến
2. Nếu có nhiều token trong file, script sẽ tự động chuyển sang token tiếp theo khi gặp rate limit
3. Dữ liệu thu thập được lưu thành các file JSON trong thư mục `data/github_commits`
4. Khi tất cả token đều hết rate limit, script sẽ:
   - Hiển thị thông báo về thời điểm reset rate limit
   - Tự động đợi đến thời điểm đó
   - Tiếp tục thu thập từ vị trí đã dừng

### Giám sát quá trình

- Kiểm tra file log `auto_collect.log` để theo dõi tiến trình
- Trạng thái thu thập được lưu trong `data/github_commits/collection_state.json`

## Thu thập thủ công

Nếu bạn muốn kiểm soát quá trình thu thập một cách thủ công:

```
python collect_100k.py collect --token_file ../../data/tokens.txt --output_dir ../../data/github_commits --target 100000
```

## Kết hợp dữ liệu sau khi thu thập

Sau khi thu thập xong, bạn có thể kết hợp các file thành một file duy nhất:

```
python collect_100k.py merge --input_dir ../../data/github_commits --output_file ../../data/all_github_commits.json
```

## Lưu ý

- Thu thập 100k commit có thể mất nhiều ngày do giới hạn rate limit của GitHub API
- Mỗi token chỉ có 5000 request/giờ
- Việc sử dụng nhiều token sẽ giúp tăng tốc quá trình thu thập
- Quá trình tự động có thể bị ngắt do mất kết nối mạng, tắt máy tính, v.v.
- Trong trường hợp ngắt quá trình, bạn có thể chạy lại lệnh `python auto_collect.py` để tiếp tục
