# Cấu trúc Component RepositoryMembers (Sau Refactor)

Tài liệu này mô tả cấu trúc và luồng logic của component `RepositoryMembers.jsx` sau khi đã được tách nhỏ thành các component con tái sử dụng.

## Tổng quan cấu trúc

- **RepositoryMembers.jsx** là component cha, chịu trách nhiệm hiển thị thành viên repository, phân tích commit và các tính năng AI. Hầu hết UI và logic được chuyển giao cho các component con.

### Các component con chính

- **ControlPanel**: Điều khiển chọn nhánh, chọn mô hình AI, bật/tắt tính năng AI.
- **OverviewStats**: Thống kê tổng quan về thành viên và nhánh.
- **AIFeaturesPanel**: Hiển thị trạng thái mô hình AI và các thông tin liên quan (có thể bật/tắt).
- **MemberList**: Danh sách thành viên (sidebar), cho phép chọn thành viên.
- **CommitAnalyticsPanel**: Hiển thị phân tích, thống kê commit của thành viên được chọn.
- **CommitList**: Danh sách commit có phân trang, lọc theo loại commit/lĩnh vực.
- **MultiFusionInsights**: (Tùy chọn) Hiển thị phân tích AI nâng cao nếu có cho thành viên được chọn.

## Quản lý state

- **members**: Danh sách thành viên repository.
- **selectedMember**: Thành viên đang được chọn để phân tích.
- **memberCommits**: Dữ liệu commit và phân tích của thành viên được chọn.
- **loading, analysisLoading**: Trạng thái loading khi tải thành viên hoặc phân tích.
- **showAIFeatures**: Bật/tắt panel tính năng AI.
- **useAI, aiModel, aiModelStatus, multiFusionV2Status**: Chọn mô hình AI và trạng thái mô hình.
- **branches, selectedBranch, branchesLoading**: Danh sách nhánh, nhánh đang chọn, trạng thái loading nhánh.
- **commitTypeFilter, techAreaFilter, currentPage, pageSize**: Lọc và phân trang danh sách commit.

## Luồng render

1. **Chưa chọn repository**: Hiển thị trạng thái trống.
2. **Header**: Hiển thị tên repository và `ControlPanel`.
3. **OverviewStats**: Thống kê tổng quan repository.
4. **AIFeaturesPanel**: (Tùy chọn) Hiển thị trạng thái mô hình AI nếu được bật.
5. **Nội dung chính (Row)**:
   - **Bên trái (Col)**: `MemberList` để chọn thành viên.
   - **Bên phải (Col)**: Nếu đã chọn thành viên, hiển thị:
     - `CommitAnalyticsPanel` (phân tích commit)
     - `CommitList` (danh sách commit, có lọc/phân trang nếu có commit)
6. **MultiFusionInsights**: (Tùy chọn) Hiển thị nếu có phân tích AI nâng cao cho thành viên.

## Luồng dữ liệu

- Tất cả việc lấy dữ liệu (thành viên, nhánh, trạng thái AI, phân tích commit) đều thực hiện ở component cha và truyền xuống các component con qua props.
- Component con chỉ chịu trách nhiệm hiển thị và logic UI.
- Việc lọc, phân trang, chọn thành viên đều quản lý ở cha và truyền xuống qua props.

## Lợi ích của refactor

- **Tách biệt trách nhiệm**: Mỗi khối UI là một component riêng biệt.
- **Dễ bảo trì**: Logic và UI từng tính năng được cô lập.
- **Tái sử dụng**: Component con có thể dùng lại ở nơi khác nếu cần.
- **File cha gọn gàng**: `RepositoryMembers.jsx` ngắn gọn, dễ đọc hơn nhiều.

---

**Cập nhật lần cuối:** 6/7/2025
