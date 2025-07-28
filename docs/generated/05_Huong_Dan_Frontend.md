# Hướng dẫn Frontend

_Tài liệu này cung cấp hướng dẫn chi tiết về cách cài đặt, cấu trúc và các thành phần chính của ứng dụng Frontend._

## 1. Tổng quan

Frontend là một ứng dụng trang đơn (Single-Page Application - SPA) được xây dựng bằng **React** và công cụ build **Vite**. Nó chịu trách nhiệm hiển thị giao diện người dùng, tương tác với người dùng và giao tiếp với Backend qua API để lấy dữ liệu và hiển thị các phân tích.

## 2. Công nghệ sử dụng

- **Framework**: React 18
- **Build Tool**: Vite
- **Ngôn ngữ**: JavaScript (JSX)
- **Quản lý định tuyến (Routing)**: `react-router-dom`
- **Thư viện UI**: Ant Design (`antd`), kết hợp với Tailwind CSS để tùy chỉnh giao diện.
- **Gọi API**: `axios`
- **Biểu đồ & Trực quan hóa**: `recharts`, `chart.js`
- **Kéo và thả (Drag and Drop)**: `@dnd-kit`

## 3. Hướng dẫn cài đặt

1.  **Di chuyển đến thư mục Frontend**
    ```bash
    cd frontend/
    ```

2.  **Cài đặt các gói phụ thuộc**
    Sử dụng `npm` (hoặc `yarn`) để cài đặt tất cả các thư viện cần thiết được định nghĩa trong `package.json`.
    ```bash
    npm install
    ```

3.  **Cấu hình biến môi trường**
    Tạo một file `.env.local` trong thư mục `frontend/`. File này sẽ chứa các biến môi trường dành riêng cho môi trường phát triển của bạn.

    **Biến quan trọng nhất:**
    ```dotenv
    # Địa chỉ URL của server backend
    VITE_API_URL=http://127.0.0.1:8000
    ```
    `Vite` yêu cầu các biến môi trường phía client phải có tiền tố `VITE_`. Biến này cho phép ứng dụng frontend biết nơi để gửi các yêu cầu API.

4.  **Khởi động server phát triển**
    Chạy lệnh sau để khởi động server phát triển với tính năng hot-reload.
    ```bash
    npm run dev
    ```
    Ứng dụng sẽ có sẵn tại `http://localhost:5173` (hoặc một cổng khác nếu cổng 5173 đã được sử dụng).

5.  **Build ứng dụng cho Production**
    Để tạo một phiên bản tối ưu hóa của ứng dụng cho môi trường production, chạy lệnh:
    ```bash
    npm run build
    ```
    Các file tĩnh sẽ được tạo ra trong thư mục `dist/`.

## 4. Cấu trúc thư mục Frontend

Cấu trúc thư mục trong `frontend/src/` được tổ chức một cách khoa học để dễ dàng quản lý và mở rộng:

-   **`main.jsx`**: Điểm khởi đầu (entry point) của ứng dụng React. Đây là nơi ứng dụng được gắn vào DOM.
-   **`App.jsx`**: Component gốc của ứng dụng, nơi định nghĩa cấu trúc layout chính và các router.
-   **`api/`**: Chứa các hàm chịu trách nhiệm giao tiếp với backend API. Việc tập trung logic gọi API ở đây giúp dễ dàng quản lý và tái sử dụng.
-   **`assets/`**: Chứa các tài nguyên tĩnh như hình ảnh, icon, font chữ.
-   **`components/`**: Chứa các **component UI có thể tái sử dụng** trên nhiều trang khác nhau (ví dụ: `Button`, `Card`, `Header`, `Sidebar`).
-   **`contexts/`**: Chứa các React Context để quản lý trạng thái toàn cục (global state). Ví dụ, `AuthContext` có thể được sử dụng để lưu trữ thông tin đăng nhập của người dùng và chia sẻ nó cho toàn bộ ứng dụng.
-   **`features/`**: Chứa các file liên quan đến một tính năng cụ thể. Ví dụ, thư mục `features/commit-analysis` có thể chứa tất cả các component, hook và service liên quan đến việc hiển thị phân tích commit.
-   **`hooks/`**: Chứa các **custom hook** (ví dụ: `useFetch`, `useAuth`). Custom hook giúp tái sử dụng logic có trạng thái (stateful logic) giữa các component.
-   **`pages/`**: Chứa các component đóng vai trò là một trang hoàn chỉnh trong ứng dụng (ví dụ: `HomePage`, `LoginPage`, `DashboardPage`). Các component này thường kết hợp nhiều component nhỏ hơn từ `components/`.
-   **`routes/`**: Định nghĩa cấu hình định tuyến cho ứng dụng bằng `react-router-dom`, ánh xạ các URL đến các component trang tương ứng.
-   **`services/`**: Có thể chứa logic nghiệp vụ phức tạp phía client hoặc các lớp tiện ích không trực tiếp liên quan đến UI.
-   **`utils/`**: Chứa các hàm tiện ích chung có thể được sử dụng ở bất kỳ đâu trong ứng dụng (ví dụ: hàm định dạng ngày tháng, hàm tính toán).
