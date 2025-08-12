import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { useAuth } from '@components/layout/useAuth';
import { ROUTES, MESSAGES } from '@constants/auth';

/**
 * Component cho các route public - redirect nếu đã xác thực
 * @param {Object} props
 * @param {React.ReactNode} props.children - Component con sẽ được render nếu chưa xác thực
 * @param {string} props.redirectTo - Đường dẫn redirect nếu đã xác thực (mặc định: '/dashboard')
 * @returns {React.ReactNode}
 */
const PublicRoute = ({ children, redirectTo = ROUTES.PROTECTED.DASHBOARD }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  // Hiển thị loading khi đang kiểm tra trạng thái xác thực
  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh'
      }}>
        <Spin size="large" tip={MESSAGES.LOADING.CHECKING_AUTH} />
      </div>
    );
  }

  // Redirect đến dashboard nếu đã xác thực
  if (isAuthenticated) {
    // Lấy URL từ state hoặc redirect đến trang mặc định
    const from = location.state?.from?.pathname || redirectTo;
    return <Navigate to={from} replace />;
  }

  // Render component con nếu chưa xác thực
  return children;
};

export default PublicRoute;
