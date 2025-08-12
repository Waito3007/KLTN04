import React, { useState, useEffect } from 'react';
import { Layout, ConfigProvider, App, Typography, Space, Divider } from 'antd';
import { GithubOutlined, HeartFilled, RocketOutlined } from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';
import AppSidebar from './AppSidebar';
import { useAuth } from './useAuth';
import { Toast } from '@components/common';
import { ROUTES, MESSAGES } from '@constants/auth';

const { Content, Footer } = Layout;
const { Text, Link } = Typography;

const AppLayout = ({ children, showSidebar = true }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  // Các trang không hiển thị sidebar
  const noSidebarPages = [ROUTES.PUBLIC.LOGIN, ROUTES.PUBLIC.AUTH_SUCCESS, ROUTES.PUBLIC.HOME];

  // Kiểm tra xem có nên hiển thị sidebar không
  const shouldShowSidebar = showSidebar && !noSidebarPages.includes(location.pathname);

  // Toggle sidebar
  const handleSidebarToggle = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  // Logout handler - đơn giản hóa vì logic đã được xử lý trong AuthContext
  const handleLogout = () => {
    try {
      // Gọi logout từ AuthContext (đã xử lý localStorage cleanup)
      logout();
      
      // Show toast
      Toast.success(MESSAGES.AUTH.LOGOUT_SUCCESS);
      
      // Navigate to login
      navigate(ROUTES.PUBLIC.LOGIN);
    } catch (error) {
      console.error('Lỗi khi đăng xuất:', error);
      Toast.error(MESSAGES.AUTH.LOGOUT_ERROR);
    }
  };

  // Responsive sidebar collapse
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarCollapsed(true);
      } else if (window.innerWidth > 1200) {
        setSidebarCollapsed(false);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Check initial size

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const layoutStyle = {
    minHeight: '100vh',
    background: '#f0f2f5',
    display: 'flex',
    flexDirection: 'column'
  };

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 8,
        },
      }}
    >
      <App>
        <Layout style={layoutStyle}>
          {/* Main content area - sẽ chứa cả content và footer */}
          <div style={{
            marginLeft: shouldShowSidebar ? (sidebarCollapsed ? 80 : 260) : 0,
            transition: 'margin-left 0.2s',
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh'
          }}>
            <Content style={{
              flex: 1,
              background: 'transparent',
              padding: '20px',
              overflow: 'auto'
            }}>
              {children}
            </Content>

            {/* Footer - Sticky at bottom */}
            <Footer style={{
              textAlign: 'center',
              padding: '20px 40px',
              background: '#001529', // Dark theme to match sidebar
              borderTop: '1px solid #303030',
              position: 'relative',
              zIndex: 1,
              flexShrink: 0 // Ngăn footer bị co lại
            }}>
              <Space direction="vertical" size="small">
                <Space size="large" wrap>
                  <Space>
                    <RocketOutlined style={{ color: '#1890ff' }} />
                    <Text style={{ color: 'rgba(255, 255, 255, 0.85)', fontWeight: '500' }}>
                      KLTN04 - AI Project Management
                    </Text>
                  </Space>
                  
                  <Divider type="vertical" style={{ borderColor: 'rgba(255, 255, 255, 0.2)' }} />
                  
                  <Space>
                    <Text style={{ color: 'rgba(255, 255, 255, 0.65)' }}>
                      Made with
                    </Text>
                    <HeartFilled style={{ color: '#ff4d4f' }} />
                    <Text style={{ color: 'rgba(255, 255, 255, 0.65)' }}>
                      by SangVu, NghiaLe
                    </Text>
                  </Space>
                  
                  <Divider type="vertical" style={{ borderColor: 'rgba(255, 255, 255, 0.2)' }} />
                  
                  <Link 
                    href="https://github.com/Waito3007/KLTN04" 
                    target="_blank"
                    style={{ 
                      color: '#1890ff',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                  >
                    <GithubOutlined />
                    GitHub Repository
                  </Link>
                </Space>
                
                <Text style={{ 
                  color: 'rgba(255, 255, 255, 0.45)',
                  fontSize: '12px'
                }}>
                  © 2025 KLTN04 Project. Powered by React + FastAPI + AI
                </Text>
              </Space>
            </Footer>
          </div>

          {/* Sidebar - render sau để nó nằm trên top */}
          {shouldShowSidebar && (
            <AppSidebar
              collapsed={sidebarCollapsed}
              onToggle={handleSidebarToggle}
              user={user}
              onLogout={handleLogout}
              theme="dark"
            />
          )}
        </Layout>
      </App>
    </ConfigProvider>
  );
};

export default AppLayout;
