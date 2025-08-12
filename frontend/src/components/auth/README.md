# Hệ thống Xác thực React (AuthContext)

## Tổng quan

Hệ thống xác thực đã được tái cấu trúc để giải quyết các vấn đề sau:
- **Prop Drilling**: Không còn cần truyền props user qua nhiều cấp component
- **Khó mở rộng**: Logic xác thực được tách biệt khỏi UI components
- **Hiệu suất**: Tối ưu hóa việc đọc localStorage và re-render

## Cấu trúc thư mục

```
src/
├── contexts/
│   └── AuthContext.js          # Context definition
├── components/
│   ├── auth/
│   │   ├── ProtectedRoute.jsx  # Bảo vệ route cần xác thực
│   │   └── PublicRoute.jsx     # Route cho user chưa đăng nhập
│   └── layout/
│       ├── AuthContext.jsx     # AuthProvider component
│       ├── useAuth.js          # Custom hook
│       └── index.js            # Export tập trung
├── constants/
│   └── auth.js                 # Hằng số và cấu hình
└── App.jsx                     # Root component with AuthProvider
```

## Cách sử dụng

### 1. Cài đặt AuthProvider (Đã setup sẵn)

```jsx
// App.jsx
import { AuthProvider } from "@components/layout/AuthContext";

function App() {
  return (
    <AuthProvider>
      {/* Toàn bộ app */}
    </AuthProvider>
  );
}
```

### 2. Sử dụng useAuth hook trong components

```jsx
import { useAuth } from '@components/layout/useAuth';

function MyComponent() {
  const { 
    user,           // Thông tin user hiện tại
    isLoading,      // Trạng thái loading
    isAuthenticated,// Trạng thái đã xác thực
    login,          // Hàm đăng nhập
    logout,         // Hàm đăng xuất
    updateUser      // Hàm cập nhật thông tin user
  } = useAuth();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <div>Please login</div>;
  }

  return (
    <div>
      <h1>Welcome, {user.name}!</h1>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

### 3. Bảo vệ routes

#### Protected Routes (Cần đăng nhập)
```jsx
import ProtectedRoute from '@components/auth/ProtectedRoute';

<Route path="/dashboard" element={
  <ProtectedRoute>
    <Dashboard />
  </ProtectedRoute>
} />
```

#### Public Routes (Chỉ cho user chưa đăng nhập)
```jsx
import PublicRoute from '@components/auth/PublicRoute';

<Route path="/login" element={
  <PublicRoute>
    <Login />
  </PublicRoute>
} />
```

### 4. Xử lý đăng nhập

```jsx
function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (userData) => {
    try {
      login(userData);
      navigate('/dashboard');
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  // ... rest of component
}
```

## API Reference

### AuthContext

| Thuộc tính | Kiểu | Mô tả |
|------------|------|-------|
| `user` | `Object \| null` | Thông tin người dùng hiện tại |
| `isLoading` | `boolean` | Trạng thái đang kiểm tra xác thực |
| `isAuthenticated` | `boolean` | Trạng thái đã xác thực |
| `login(userData)` | `function` | Hàm đăng nhập |
| `logout()` | `function` | Hàm đăng xuất |
| `updateUser(data)` | `function` | Hàm cập nhật thông tin user |

### useAuth Hook

```javascript
const {
  user,
  isLoading, 
  isAuthenticated,
  login,
  logout,
  updateUser
} = useAuth();
```

**Lưu ý**: Hook này chỉ có thể sử dụng bên trong `AuthProvider`.

### ProtectedRoute

```jsx
<ProtectedRoute redirectTo="/login">
  <ComponentCanXacThuc />
</ProtectedRoute>
```

### PublicRoute

```jsx
<PublicRoute redirectTo="/dashboard">
  <ComponentPublic />
</PublicRoute>
```

## Hằng số và cấu hình

### STORAGE_KEYS
```javascript
import { STORAGE_KEYS } from '@constants/auth';

// STORAGE_KEYS.GITHUB_PROFILE
// STORAGE_KEYS.USER  
// STORAGE_KEYS.ACCESS_TOKEN
```

### ROUTES
```javascript
import { ROUTES } from '@constants/auth';

// ROUTES.PUBLIC.LOGIN
// ROUTES.PUBLIC.HOME
// ROUTES.PROTECTED.DASHBOARD
// etc...
```

### MESSAGES
```javascript
import { MESSAGES } from '@constants/auth';

// MESSAGES.AUTH.LOGIN_SUCCESS
// MESSAGES.AUTH.LOGOUT_SUCCESS
// MESSAGES.LOADING.CHECKING_AUTH
// etc...
```

## Tối ưu hóa hiệu suất

1. **Memoization**: Context value được memoize để tránh re-render không cần thiết
2. **Lazy loading**: Components được lazy load để giảm bundle size
3. **Error boundaries**: Xử lý lỗi gracefully
4. **Loading states**: UI feedback trong quá trình xác thực

## Xử lý lỗi

Hệ thống tự động xử lý các trường hợp lỗi:
- localStorage không khả dụng
- Dữ liệu user bị lỗi/hỏng
- Network errors
- Invalid user data

## Migration từ hệ thống cũ

Nếu bạn đang sử dụng hệ thống xác thực cũ:

1. Thay thế `useState` user bằng `useAuth()` hook
2. Loại bỏ logic localStorage trong components
3. Wrap routes với `ProtectedRoute` hoặc `PublicRoute`
4. Sử dụng constants thay vì hardcode strings

## Troubleshooting

### Lỗi: "useAuth must be used within an AuthProvider"
- Đảm bảo component sử dụng `useAuth` được wrap trong `AuthProvider`

### User state không được persist
- Kiểm tra localStorage có bị disable không
- Xem console có lỗi JSON parse không

### Route protection không hoạt động
- Đảm bảo đã wrap route với `ProtectedRoute` hoặc `PublicRoute`
- Kiểm tra `isAuthenticated` state

## Best Practices

1. **Luôn sử dụng constants** thay vì hardcode strings
2. **Xử lý loading states** để UX tốt hơn  
3. **Error handling** ở tất cả authentication flows
4. **Không truy cập localStorage trực tiếp** - sử dụng AuthContext
5. **Sử dụng TypeScript** để type safety (có thể thêm sau)
