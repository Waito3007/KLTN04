# Common Components Guide

Hướng dẫn sử dụng các common components để đảm bảo tính nhất quán trong toàn bộ ứng dụng.

## 🚀 Quick Start

```jsx
import { Loading, Modal, Toast, EmptyState } from "@components/common";

// Sử dụng trong component
<Loading variant="gradient" text="Đang tải..." />;
```

## 📦 Available Components

### 1. Loading Component

Component loading với nhiều variant khác nhau.

```jsx
import { Loading } from '@components/common';

// Basic loading
<Loading text="Đang tải..." />

// Fullscreen loading
<Loading variant="fullscreen" text="Đang xử lý..." />

// Gradient background loading (như trang login)
<Loading variant="gradient" text="Đang tải dashboard..." size="large" />

// Overlay loading
<Loading variant="overlay" text="Đang lưu..." />

// Inline loading
<Loading variant="inline" size="small" />

// Wrapper loading
<Loading spinning={isLoading} text="Loading...">
  <YourComponent />
</Loading>
```

**Props:**

- `variant`: 'default' | 'fullscreen' | 'gradient' | 'overlay' | 'inline'
- `size`: 'small' | 'default' | 'large'
- `text`: string
- `spinning`: boolean
- `children`: ReactNode

### 2. Modal Component

Modal hiện đại với thiết kế gradient.

```jsx
import { Modal } from "@components/common";

<Modal
  visible={modalVisible}
  onClose={() => setModalVisible(false)}
  onConfirm={handleConfirm}
  title="Xác nhận"
  content="Bạn có chắc chắn muốn thực hiện hành động này?"
  type="warning"
  size="default"
/>;

// Quick modals
Modal.confirm({
  title: "Xác nhận xóa",
  content: "Bạn có chắc chắn muốn xóa item này?",
  onOk: handleDelete,
});

Modal.success({
  title: "Thành công",
  content: "Dữ liệu đã được lưu thành công!",
});
```

**Props:**

- `visible`: boolean
- `type`: 'default' | 'confirm' | 'info' | 'success' | 'warning' | 'danger'
- `size`: 'small' | 'default' | 'large'
- `title`: string
- `content`: string | ReactNode
- `onClose`: function
- `onConfirm`: function

### 3. Drawer Component

Drawer với thiết kế gradient header.

```jsx
import { Drawer } from "@components/common";

<Drawer
  visible={drawerVisible}
  onClose={() => setDrawerVisible(false)}
  title="Chi tiết"
  placement="right"
  size="large"
  showFooter
  footerActions={<Button onClick={handleSave}>Lưu</Button>}
>
  <YourContent />
</Drawer>;
```

**Props:**

- `visible`: boolean
- `placement`: 'left' | 'right' | 'top' | 'bottom'
- `size`: 'small' | 'default' | 'large'
- `title`: string
- `showFooter`: boolean
- `footerActions`: ReactNode

### 4. Toast Component

Thông báo nhất quán cho toàn ứng dụng.

```jsx
import { Toast } from "@components/common";

// Message toasts (ngắn gọn)
Toast.success("Lưu thành công!");
Toast.error("Có lỗi xảy ra!");
Toast.warning("Cảnh báo!");
Toast.info("Thông tin");
Toast.loading("Đang xử lý...", 0); // duration = 0 = không tự đóng

// Notification toasts (chi tiết hơn)
Toast.notify.success({
  message: "Thành công!",
  description: "Dữ liệu đã được lưu thành công.",
  duration: 4.5,
});

Toast.notify.error({
  message: "Lỗi!",
  description: "Không thể kết nối đến server.",
  duration: 6,
});

// Progress notification
const progressKey = Toast.notify.progress({
  message: "Đang đồng bộ",
  description: "Đang đồng bộ dữ liệu...",
});

// Cập nhật progress
Toast.notify.info({
  key: progressKey,
  message: "Hoàn thành",
  description: "Đồng bộ thành công!",
});
```

### 5. EmptyState Component

Hiển thị trạng thái trống với các variant khác nhau.

```jsx
import { EmptyState } from '@components/common';

// No data
<EmptyState
  type="no-data"
  action="Thêm dữ liệu"
  onAction={handleAdd}
/>

// No search results
<EmptyState
  type="no-search"
  title="Không tìm thấy kết quả"
  description="Thử tìm kiếm với từ khóa khác"
/>

// Error state
<EmptyState
  type="error"
  action="Thử lại"
  onAction={handleRetry}
/>

// Custom empty state
<EmptyState
  title="Chưa có dự án"
  description="Tạo dự án đầu tiên của bạn"
  action="Tạo dự án"
  onAction={handleCreateProject}
  size="large"
/>
```

**Props:**

- `type`: 'default' | 'no-data' | 'no-search' | 'no-connection' | 'error' | 'folder'
- `size`: 'small' | 'default' | 'large'
- `title`: string
- `description`: string
- `action`: string
- `onAction`: function

### 6. ErrorBoundary Component

Bắt lỗi React và hiển thị UI fallback.

```jsx
import { ErrorBoundary } from '@components/common';

// Wrap toàn bộ app
<ErrorBoundary>
  <App />
</ErrorBoundary>

// Wrap specific component
<ErrorBoundary
  title="Lỗi tại component này"
  subTitle="Vui lòng thử lại hoặc liên hệ hỗ trợ"
>
  <RiskyComponent />
</ErrorBoundary>
```

### 7. PageLayout Component

Layout wrapper cho các trang.

```jsx
import { PageLayout } from "@components/common";

<PageLayout
  title="Quản lý dự án"
  breadcrumb={[{ title: "Dự án" }, { title: "Chi tiết" }]}
  background="gradient"
  centered
  maxWidth="1200px"
>
  <YourPageContent />
</PageLayout>;
```

**Props:**

- `title`: string
- `breadcrumb`: Array<{ title: string, href?: string }>
- `background`: 'default' | 'gradient' | 'transparent'
- `padding`: 'none' | 'small' | 'default' | 'large'
- `centered`: boolean
- `maxWidth`: string

### 8. SearchBox Component

Tìm kiếm với filter nâng cao.

```jsx
import { SearchBox } from "@components/common";

const filters = [
  {
    key: "status",
    label: "Trạng thái",
    options: [
      { value: "active", label: "Hoạt động" },
      { value: "inactive", label: "Không hoạt động" },
    ],
  },
];

<SearchBox
  placeholder="Tìm kiếm dự án..."
  onSearch={handleSearch}
  onFilter={handleFilter}
  filters={filters}
  activeFilters={activeFilters}
  debounceMs={500}
  clearable
/>;
```

## 🎨 Design System

Tất cả components sử dụng:

- **Colors**: Gradient themes (#667eea, #764ba2)
- **Border radius**: 8px, 12px, 16px cho các kích thước khác nhau
- **Shadows**: Box shadows với opacity thấp
- **Typography**: Ant Design typography với custom weights
- **Spacing**: 8px, 16px, 24px, 32px grid system

## 🔄 Migration Guide

### Thay thế Ant Design components cũ:

```jsx
// Cũ
import { message, Modal, Spin } from 'antd';
message.success('Success!');
Modal.confirm({...});
<Spin tip="Loading..." />

// Mới
import { Toast, Modal, Loading } from '@components/common';
Toast.success('Success!');
Modal.confirm({...});
<Loading text="Loading..." />
```

## 📱 Responsive Design

Tất cả components được thiết kế responsive:

- Mobile: < 768px
- Tablet: 768px - 1200px
- Desktop: > 1200px

## 🧪 Demo Page

Xem demo tại: `/demo`

Hoặc:

```jsx
import ComponentDemo from "@pages/ComponentDemo";
```

## 📝 Best Practices

1. **Sử dụng Loading thay vì Spin trực tiếp**
2. **Dùng Toast thay vì message/notification**
3. **Wrap pages với PageLayout**
4. **Sử dụng EmptyState cho trạng thái trống**
5. **Wrap risky components với ErrorBoundary**
6. **Sử dụng SearchBox cho tìm kiếm có filter**

## 🎯 Consistency Rules

- Tất cả modals phải có gradient header
- Loading states phải có text descriptions
- Empty states phải có action buttons khi có thể
- Error messages phải rõ ràng và actionable
- Toast messages phải ngắn gọn, notifications chi tiết hơn
