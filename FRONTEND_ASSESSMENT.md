# Đánh giá và Đề xuất Tối ưu Frontend

Tài liệu này tổng hợp các vấn đề chính được phát hiện trong quá trình rà soát mã nguồn frontend. Các vấn đề được sắp xếp theo mức độ ưu tiên để tập trung vào những thay đổi có ảnh hưởng lớn nhất trước khi triển khai (deployment).

---

## Mức độ Nghiêm trọng (P0 - Critical)

*Đây là những vấn đề bắt buộc phải khắc phục vì chúng ảnh hưởng trực tiếp và nghiêm trọng đến hiệu suất và trải nghiệm người dùng.*

### 1. Không áp dụng Code-Splitting cho các Route
- **Vấn đề:** Toàn bộ mã nguồn của tất cả các trang (`Login`, `Dashboard`, `RepositoryList`, v.v.) được gộp vào một tệp duy nhất và tải ngay từ đầu.
- **Tác động:** Thời gian tải trang lần đầu (Initial Page Load) cực kỳ chậm, gây trải nghiệm tồi tệ cho người dùng, đặc biệt trên các kết nối mạng yếu.
- **Vị trí:** `frontend/src/App.jsx`
- **Đề xuất:**
    - Sử dụng `React.lazy()` để import động các component của trang.
    - Bọc `<Routes>` trong component `<React.Suspense>` với một fallback UI (ví dụ: spinner loading) để xử lý trạng thái chờ tải.

**Ví dụ:**
```jsx
// Trước khi sửa
import DashboardModern from "@pages/DashboardModern";

// Sau khi sửa
const DashboardModern = React.lazy(() => import("@pages/DashboardModern"));

// trong JSX
<Suspense fallback={<Loading />}>
  <Routes>
    <Route path="/dashboard" element={<DashboardModern />} />
    {/* ... các route khác */}
  </Routes>
</Suspense>
```

---

## Mức độ Cao (P1 - High)

*Các vấn đề ảnh hưởng lớn đến khả năng bảo trì, mở rộng và hiệu suất tổng thể của ứng dụng.*

### 1. Chiến lược Styling hỗn loạn và không hiệu quả
- **Vấn đề:** Dự án đang sử dụng đồng thời 3-4 phương pháp styling khác nhau:
    1.  Hệ thống component và style của Ant Design.
    2.  Inline styles được tính toán bằng JavaScript trong hầu hết component tùy chỉnh.
    3.  Các tệp `.css` toàn cục với việc ghi đè style bằng `!important`.
    4.  Sự tồn tại của `styled-components` và `tailwindcss` trong `package.json`.
- **Tác động:**
    - **Bundle size lớn:** Phải tải nhiều thư viện styling không cần thiết.
    - **Khó bảo trì:** Việc debug và thay đổi UI rất phức tạp.
    - **Xung đột CSS:** Phải dùng `!important` để ghi đè style, một thực hành nên tránh.
- **Vị trí:** Toàn bộ dự án (`AppLayout.jsx`, `components/common/`, `components/Dashboard/...`).
- **Đề xuất:**
    - **Chọn MỘT chiến lược chính.** Khuyến nghị: **Giữ lại Ant Design làm nền tảng.**
    - Tận dụng `<ConfigProvider>` của `antd` để tùy chỉnh theme (màu sắc, bo góc, v.v.) một cách toàn cục.
    - Đối với các tùy chỉnh phức tạp hơn cho từng component, hãy sử dụng **CSS Modules** (`*.module.css`) thay vì inline styles. CSS Modules giúp đóng gói style theo component và tránh xung đột.
    - Lên kế hoạch loại bỏ dần các file `.css` toàn cục và việc sử dụng inline styles.

### 2. Quản lý Trạng thái Toàn cục (Authentication) không hiệu quả
- **Vấn đề:** Trạng thái người dùng được quản lý bằng `useState` và `localStorage` ngay trong component `AppLayout`. Logic này được chạy lại trên mỗi lần chuyển trang.
- **Tác động:**
    - **Prop Drilling:** Khó chia sẻ trạng thái người dùng cho các component con nằm sâu bên trong.
    - **Khó mở rộng:** Logic xác thực bị ràng buộc chặt chẽ vào một component giao diện.
    - **Hiệu suất:** Đọc `localStorage` một cách không cần thiết trên mỗi lần điều hướng.
- **Vị trí:** `frontend/src/components/layout/AppLayout.jsx`
- **Đề xuất:**
    - Tái cấu trúc bằng cách tạo ra một `AuthContext`.
    - `AuthProvider` sẽ bao bọc toàn bộ ứng dụng, chứa logic lấy/lưu dữ liệu người dùng và cung cấp thông tin này cho toàn bộ cây component thông qua một hook `useAuth()`.

---

## Mức độ Trung bình (P2 - Medium)

*Các vấn đề ảnh hưởng đến chất lượng mã nguồn và có thể gây ra lỗi hoặc làm giảm hiệu suất trong các trường hợp cụ thể.*

### 1. Tính toán lại không cần thiết trong Component
- **Vấn đề:** Các logic lọc và xử lý mảng dữ liệu lớn được thực thi lại trong mỗi lần render của component, ngay cả khi dữ liệu không thay đổi.
- **Tác động:** Lãng phí tài nguyên CPU, có thể gây giật, lag trong các component phức tạp.
- **Vị trí:** `frontend/src/components/Dashboard/CommitAnalyst/BranchCommitAnalysis.jsx`
- **Đề xuất:**
    - Sử dụng `React.useMemo` để "ghi nhớ" kết quả của các phép tính toán nặng (ví dụ: `typeCount`, `filtered`). `useMemo` sẽ chỉ tính toán lại khi các phụ thuộc của nó thay đổi.

### 2. Tồn tại Code lỗi / Code thừa
- **Vấn đề:** Route `path="/repo-details"` trỏ đến component `<RepoDetails />` mà không có tham số URL cần thiết, có khả năng gây lỗi.
- **Tác động:** Gây ra lỗi tiềm ẩn và làm mã nguồn khó hiểu.
- **Vị trí:** `frontend/src/App.jsx`
- **Đề xuất:** Xóa bỏ route `/repo-details` nếu nó không được sử dụng hoặc sửa lại cho đúng.

---

## Mức độ Thấp (P3 - Low)

*Các cải tiến nhỏ giúp mã nguồn sạch hơn và dễ bảo trì hơn.*

### 1. Dọn dẹp mã nguồn
- **Vấn đề:** Tồn tại các câu lệnh `console.log` dùng để debug trong mã nguồn.
- **Tác động:** Rò rỉ thông tin không cần thiết ra console của trình duyệt ở môi trường production.
- **Vị trí:** `BranchCommitAnalysis.jsx` và có thể ở nhiều nơi khác.
- **Đề xuất:** Xóa tất cả các `console.log` thủ công hoặc cấu hình công cụ build (Vite/Terser) để tự động loại bỏ chúng khi build cho production.

### 2. Hardcode các Hằng số
- **Vấn đề:** Các key của `localStorage` (ví dụ: `'github_profile'`) được viết trực tiếp dưới dạng chuỗi.
- **Tác động:** Dễ gõ nhầm, khó thay đổi đồng bộ.
- **Vị trí:** `frontend/src/components/layout/AppLayout.jsx`
- **Đề xuất:** Tạo một tệp hằng số (ví dụ: `src/config.js`) để lưu trữ các giá trị này và import khi cần.

### 3. Xử lý sự kiện chưa tối ưu
- **Vấn đề:** Sự kiện `resize` trong `AppLayout` được xử lý trực tiếp, có thể kích hoạt render nhiều lần.
- **Tác động:** Ảnh hưởng hiệu suất không đáng kể nhưng là một điểm có thể cải thiện.
- **Vị trí:** `frontend/src/components/layout/AppLayout.jsx`
- **Đề xuất:** Sử dụng kỹ thuật "debouncing" cho hàm xử lý sự kiện `resize` để giảm số lần thực thi không cần thiết.
