import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { useAuth } from '@components/layout/useAuth';
import { ROUTES, MESSAGES } from '@constants/auth';

/**
 * Component bảo vệ route - chỉ cho phép truy cập khi đã xác thực
 * @param {Object} props
 * @param {React.ReactNode} props.children - Component con sẽ được render nếu đã xác thực
 * @param {string} props.redirectTo - Đường dẫn redirect nếu chưa xác thực (mặc định: '/login')
 * @returns {React.ReactNode}
 */
const ProtectedRoute = ({ children, redirectTo = ROUTES.PUBLIC.LOGIN }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  // Hiển thị loading khi đang kiểm tra trạng thái xác thực
  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        position: 'relative',
      }}>
        <div style={{ position: 'absolute', zIndex: 1 }}>
          <Spin size="large" tip={MESSAGES.LOADING.CHECKING_AUTH} />
        </div>
      </div>
    );
  }

  // Redirect đến trang login nếu chưa xác thực
  if (!isAuthenticated) {
    return (
      <Navigate 
        to={redirectTo} 
        state={{ from: location }} 
        replace 
      />
    );
  }

  // Render component con nếu đã xác thực
  return children;
};

export default ProtectedRoute;
