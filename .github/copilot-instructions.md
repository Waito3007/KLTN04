# HƯỚNG DẪN CHO GITHUB COPILOT - DỰ ÁN KLTN04 (PHIÊN BẢN CUỐI CÙNG & HOÀN CHỈNH NHẤT)

## 1. NGUYÊN TẮC CỐT LÕI VÀ MỤC TIÊU

- **Mục tiêu dự án**: Hỗ trợ dự án KLTN04, ứng dụng AI quản lý tiến độ.
- **NGÔN NGỮ BẮT BUỘC**: Mọi phản hồi, giải thích, bình luận phải bằng **TIẾNG VIỆT**.
- **NGUYÊN TẮC LÀM VIỆC**:
  - **TUYỆT ĐỐI KHÔNG LÀM BỪA**: Hiểu rõ mã nguồn và ngữ cảnh trước khi đề xuất.
  - **LUÔN CẨN TRỌNG**: Xác thực mọi thứ, xử lý lỗi triệt để.
  - **CHẤT LƯỢNG MÃ NGUỒN**: Ưu tiên mã sạch, dễ bảo trì, và có khả năng mở rộng.

---

## 2. NGUYÊN TẮC PHÒNG NGỪA BUG (BUG PREVENTION PRINCIPLES)

- **Tính Bất biến (Immutability)**:
  - **Frontend (React)**: **CẤM** thay đổi trực tiếp `state` hoặc `props`. Luôn tạo ra một đối tượng hoặc mảng mới.
  - **Backend (Python)**: Tránh thay đổi (mutate) các tham số đầu vào của hàm.
- **Lập trình Phòng thủ (Defensive Programming)**:
  - Ở đầu mỗi hàm public hoặc service, **PHẢI** kiểm tra tính hợp lệ của các tham số quan trọng. Ném ra lỗi sớm (fail-fast) nếu đầu vào không hợp lệ.
- **Tính Idempotent của API (API Idempotency)**:
  - Các endpoint `PUT`, `PATCH`, và `DELETE` phải được thiết kế để có tính idempotent.
- **Hạn chế "Escape Hatches"**:
  - Hạn chế tối đa việc sử dụng `typing.Any` (Python) hoặc `any` (TypeScript).
- **Bao quát toàn hệ thống**: Mỗi lần code, cần kiểm tra và đảm bảo rằng các thay đổi không gây lỗi cho các phần khác của hệ thống. Điều này bao gồm việc kiểm tra các module liên quan, các service, và các API endpoint.

---

## 3. QUY TẮC TUYỆT ĐỐI (ZERO-TOLERANCE RULES)

- **Định dạng Commit Message**: **BẮT BUỘC** tuân thủ theo **Conventional Commits**. (`feat:`, `fix:`, `refactor:`, etc.)
- **Kiểm tra chất lượng mã**: Mã nguồn **PHẢI** vượt qua linter và formatter.
- **Phân tích Tĩnh (Static Analysis)**: Mã nguồn Python **PHẢI** vượt qua kiểm tra của `mypy --strict` mà không có lỗi.
- **Kiểm thử (Testing)**:
  - Mọi logic nghiệp vụ trong `services/` và `utils/` **PHẢI** có Unit Test.
  - Unit test **PHẢI** bao gồm các trường hợp biên (edge cases).
- **Không "Magic Values"**: **CẤM** sử dụng các chuỗi hoặc số không có ý nghĩa rõ ràng (magic strings/numbers) trực tiếp trong code. **PHẢI** định nghĩa chúng thành các hằng số (constants) có tên gọi tường minh.

---

## 4. CÔNG NGHỆ VÀ KIẾN TRÚC

### **Backend: FastAPI (Python)**

- **Kiến trúc**: Logic nghiệp vụ phải nằm trong `services/`, không nằm trong `routes/`.
- **Thiết kế API & Response**: Tuân thủ cấu trúc response chuẩn (`{ "status": "success", "data": ... }` hoặc `{ "status": "error", "message": ... }`).
- **Database Transactions**: Mọi chuỗi thao tác ghi (create, update, delete) liên quan đến nhau **BẮT BUỘC** phải được bao bọc trong một khối transaction duy nhất để đảm bảo tính toàn vẹn dữ liệu.

### **Frontend: React (JavaScript/Vite)**

- **Kiến trúc**: Tách biệt `pages`, `components`, `services`, và `hooks`.
- **Quản lý State**: Mặc định dùng `useState`. **CHỈ** dùng `Context API` cho state toàn cục không thay đổi thường xuyên.

---

## 5. BẢO MẬT VÀ HIỆU NĂNG

### **Bảo mật (Security)**

- **Validation**: **BẮT BUỘC** dùng Pydantic models để xác thực tất cả dữ liệu từ request.
- **Dependencies**: Định kỳ chạy `npm audit` và `pip-audit`.

### **Hiệu năng (Performance)**

- **Tránh N+1 Problems**: **BẮT BUỘC** sử dụng `selectinload` hoặc `joinedload` của SQLAlchemy.
- **Tối ưu Re-render**: Sử dụng `React.memo`, `useMemo`, `useCallback` một cách chính xác.

---

## 6. KIẾN TRÚC NÂNG CAO & VẬN HÀNH

- **Quản lý Cấu hình Tập trung**: Mọi cấu hình **PHẢI** được quản lý thông qua module `core/config.py` và nạp từ biến môi trường.
- **Custom Exceptions**: **PHẢI** định nghĩa và sử dụng các lớp exception tùy chỉnh cho logic nghiệp vụ.
- **Quản lý Database Migrations**: Mọi thay đổi schema **PHẢI** được thực hiện thông qua `Alembic`.

---

## 7. ĐỘ TIN CẬY VÀ KHẢ NĂNG QUAN SÁT (RELIABILITY & OBSERVABILITY)

- **Logging có Cấu trúc (Structured Logging)**:
  - **PHẢI** sử dụng module `logging` của Python. Log phải có cấu trúc (JSON) và chứa các thông tin ngữ cảnh (request ID, user ID).
  - Phân biệt rõ các cấp độ log: `INFO`, `WARNING`, `ERROR`.
- **Khả năng quan sát (Observability)**:
  - **Metrics**: Backend phải cung cấp một endpoint (VD: `/metrics`) để lộ các chỉ số quan trọng (VD: độ trễ request, tỷ lệ lỗi) theo chuẩn Prometheus.
  - **Distributed Tracing**: Mọi request API phải có một `trace_id` duy nhất. `trace_id` này **PHẢI** được truyền đi qua tất cả các service và được đưa vào trong mọi log liên quan đến request đó.
- **Độ tin cậy & Khả năng chịu lỗi (Fault Tolerance)**:
  - **Timeouts**: Mọi lời gọi mạng ra bên ngoài **PHẢI** có một giá trị timeout được cấu hình rõ ràng.
  - **Retries & Exponential Backoff**: Đối với các lỗi tạm thời khi gọi đến một service ngoài, cần triển khai cơ chế thử lại (retry) với thời gian chờ tăng dần.
- **Triển khai an toàn (Safe Deployments)**:
  - **Feature Flags**: Các tính năng mới có ảnh hưởng lớn **PHẢI** được bao bọc bởi cơ chế feature flag để có thể bật/tắt nhanh chóng trên production mà không cần deploy lại.
