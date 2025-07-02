# Cập Nhật Hệ Thống Đồng Bộ Commit với Force Update

## Tổng Quan

Tài liệu này mô tả các thay đổi đã thực hiện để đảm bảo rằng hệ thống luôn cập nhật thông tin commit từ GitHub, ngay cả khi commit đã tồn tại trong cơ sở dữ liệu.

## Vấn Đề

- Trước đây, khi đồng bộ commit từ GitHub, hệ thống sẽ bỏ qua các commit đã tồn tại trong cơ sở dữ liệu.
- Điều này dẫn đến việc metadata nâng cao của các commit không được cập nhật khi đồng bộ lại.

## Giải Pháp

1. Cập nhật hàm `save_multiple_commits` trong `services/commit_service.py` để hỗ trợ tham số `force_update`.
2. Khi `force_update=True`, hệ thống sẽ cập nhật thông tin của các commit đã tồn tại thay vì bỏ qua chúng.
3. Cập nhật tất cả các endpoint đồng bộ để sử dụng `force_update=True` theo mặc định.

## Các Thay Đổi Chi Tiết

### 1. Trong `services/commit_service.py`

- Thêm tham số `force_update` vào hàm `save_multiple_commits`
- Sửa đổi logic để cập nhật commit đã tồn tại khi `force_update=True`
- Cải thiện logging để phân biệt giữa commit mới và commit được cập nhật

### 2. Trong `api/routes/commit.py`

Cập nhật các endpoint sau để thêm tham số `force_update=True`:

- `/github/{owner}/{repo}/sync-commits`
- `/github/{owner}/{repo}/sync-all-branches-commits`
- `/github/{owner}/{repo}/branches/{branch_name}/sync-commits`
- `/github/{owner}/{repo}/sync-enhanced`

## Cách Sử Dụng

Các endpoint đồng bộ giờ đây sẽ mặc định cập nhật tất cả commit, kể cả những commit đã tồn tại. Điều này đảm bảo rằng metadata nâng cao luôn được cập nhật.

Nếu muốn giữ hành vi cũ (bỏ qua commit đã tồn tại), có thể thiết lập `force_update=False` trong query parameter của request:

```
POST /api/github/{owner}/{repo}/sync-commits?force_update=false
```

## Lợi Ích

- Đảm bảo tất cả commit luôn được cập nhật với metadata đầy đủ nhất từ GitHub
- Cải thiện tính nhất quán của dữ liệu
- Cho phép cập nhật các trường metadata nâng cao như `files_changed`, `modified_files`, `file_types`, `modified_directories`, v.v. mà không cần xóa và thêm lại commit

## Vấn Đề Tiềm Ẩn

- Quá trình đồng bộ có thể mất nhiều thời gian hơn do cần cập nhật commit đã tồn tại
- Có thể gây ra nhiều hoạt động ghi cơ sở dữ liệu hơn

## Kiểm Tra

Để kiểm tra các thay đổi, hãy gọi một trong các endpoint đồng bộ và xác nhận rằng:

1. Các commit đã tồn tại được cập nhật với thông tin mới nhất
2. Số lượng commit được xử lý trong phản hồi bao gồm cả commit mới và commit được cập nhật
