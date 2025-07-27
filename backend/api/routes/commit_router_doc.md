# Tài liệu API Commit Router

File này mô tả các endpoint chính trong router commit của hệ thống, giúp dev và người dùng hiểu rõ chức năng từng API.

---

## 1. Truy vấn từ Database

- **Lấy commit của branch:**

  - `GET /commits/{owner}/{repo}/branches/{branch_name}/commits`
  - Trả về danh sách commit của một branch từ database.

- **Lấy tất cả commit của repo:**

  - `GET /commits/{owner}/{repo}/commits`
  - Trả về tất cả commit của repo, hỗ trợ lọc theo branch, ngày, phân trang.

- **Lấy danh sách branch với thống kê commit:**

  - `GET /commits/{owner}/{repo}/branches`
  - Trả về tất cả branch, kèm số lượng commit, ngày commit mới nhất, v.v.

- **So sánh commit giữa hai branch:**

  - `GET /commits/{owner}/{repo}/compare/{base_branch}...{head_branch}`
  - Trả về các commit có trong head branch mà không có trong base branch.

- **Lấy chi tiết một commit:**
  - `GET /commits/{sha}`
  - Trả về thông tin chi tiết của một commit.

---

## 2. Lấy dữ liệu trực tiếp từ GitHub

- **Lấy commit của branch từ GitHub:**

  - `GET /github/{owner}/{repo}/branches/{branch_name}/commits`
  - Lấy commit của một branch trực tiếp từ GitHub API.

- **Lấy commit của repo từ GitHub:**

  - `GET /github/{owner}/{repo}/commits`
  - Lấy commit của repo trực tiếp từ GitHub API, hỗ trợ lọc nâng cao.

- **Lấy commit nâng cao từ GitHub:**

  - `GET /github/{owner}/{repo}/commits/enhanced`
  - Lấy commit với metadata nâng cao (files_changed, additions, deletions, v.v.) từ GitHub.

- **Lấy metadata file của commit:**
  - `GET /github/{owner}/{repo}/commits/{sha}/files`
  - Lấy thông tin chi tiết về file của một commit từ GitHub.

---

## 3. Đồng bộ & Quản lý

- **Đồng bộ commit từ GitHub về DB:**

  - `POST /github/{owner}/{repo}/sync-commits`
  - Đồng bộ commit từ GitHub về database cho một branch.

- **Đồng bộ commit cho tất cả branch:**

  - `POST /github/{owner}/{repo}/sync-all-branches-commits`
  - Đồng bộ commit cho tất cả branch từ GitHub về database.

- **Đồng bộ commit nâng cao từ GitHub:**

  - `POST /github/{owner}/{repo}/sync-enhanced`
  - Đồng bộ commit với metadata nâng cao từ GitHub về database.

- **Kiểm tra & sửa inconsistency:**
  - `POST /commits/{owner}/{repo}/validate-commit-consistency`
  - Kiểm tra và sửa inconsistency giữa branch_id và branch_name trong commit.

---

## 4. Thống kê & Phân tích

- **Thống kê commit:**

  - `GET /github/{owner}/{repo}/commit-stats`
  - Trả về thống kê commit tổng quan cho repo.

- **Thống kê commit nâng cao:**

  - `GET /commits/{owner}/{repo}/statistics/enhanced`
  - Trả về thống kê commit nâng cao với phân tích chi tiết.

- **Phân tích xu hướng commit:**

  - `GET /commits/{owner}/{repo}/trends`
  - Phân tích xu hướng commit trong khoảng thời gian nhất định.

- **Phân tích một commit:**

  - `POST /commits/{owner}/{repo}/analyze`
  - Phân tích chi tiết một commit, kèm khuyến nghị và điểm chất lượng.

- **So sánh commit qua GitHub:**
  - `GET /github/{owner}/{repo}/compare/{base}...{head}`
  - So sánh commit giữa hai branch qua GitHub API.

---

## Ghi chú

- Các endpoint đều trả về dữ liệu dạng JSON.
- Nên dùng endpoint database cho truy vấn nhanh, endpoint GitHub cho dữ liệu mới nhất.
- Các endpoint đồng bộ giúp cập nhật dữ liệu từ GitHub về hệ thống.
- Endpoint phân tích/thống kê hỗ trợ đánh giá chất lượng và xu hướng commit.
