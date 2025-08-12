import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Layout, 
  Menu, 
  Button, 
  Avatar, 
  Typography, 
  Space, 
  Tooltip, 
  Divider,
  Badge
} from 'antd';
import {
  DashboardOutlined,
  GithubOutlined,
  SyncOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  CodeOutlined,
  ExperimentOutlined,
  BranchesOutlined,
  FileTextOutlined,
  TeamOutlined,
  BarChartOutlined,
  RocketOutlined
} from '@ant-design/icons';

const { Sider } = Layout;
const { Text } = Typography;

const AppSidebar = ({ 
  collapsed = false, 
  onToggle,
  user = null,
  onLogout,
  theme = 'dark' // 'dark' | 'light'
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [selectedKeys, setSelectedKeys] = useState([]);

  // Cập nhật selected key khi route thay đổi
  useEffect(() => {
    const path = location.pathname;
    if (path.includes('/dashboard')) {
      setSelectedKeys(['dashboard']);
    } else if (path.includes('/sync') || path.includes('/repo-sync')) {
      setSelectedKeys(['sync']);
    } else if (path.includes('/repositories')) {
      setSelectedKeys(['repositories']);
    } else if (path.includes('/analysis')) {
      setSelectedKeys(['analysis']);
    } else if (path.includes('/commits')) {
      setSelectedKeys(['analysis']);
    } else if (path.includes('/repo')) {
      setSelectedKeys(['repositories']);
    } else if (path.includes('/demo')) {
      setSelectedKeys(['demo']);
    } else if (path.includes('/test')) {
      setSelectedKeys(['test']);
    } else {
      setSelectedKeys([]);
    }
  }, [location.pathname]);

  const menuItems = [
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
      onClick: () => navigate('/dashboard')
    },
    {
      key: 'repositories',
      icon: <GithubOutlined />,
      label: 'Repositories',
      children: [
        {
          key: 'repo-list',
          icon: <BranchesOutlined />,
          label: 'Danh sách',
          onClick: () => navigate('/repositories') // Navigate to repositories page
        },
        {
          key: 'repo-details',
          icon: <FileTextOutlined />,
          label: 'Chi tiết',
          onClick: () => navigate('/repo/Waito3007/KLTN04')
        }
      ]
    },
    {
      key: 'sync',
      icon: <SyncOutlined />,
      label: 'Đồng bộ',
      children: [
        {
          key: 'sync-page',
          icon: <SyncOutlined />,
          label: 'Sync Tool',
          onClick: () => navigate('/sync')
        },
        {
          key: 'repo-sync',
          icon: <GithubOutlined />,
          label: 'Repo Manager',
          onClick: () => navigate('/repo-sync')
        }
      ]
    },
    {
      type: 'divider'
    },
    {
      key: 'analysis',
      icon: <BarChartOutlined />,
      label: 'Phân tích',
      children: [
        {
          key: 'repo-analysis',
          icon: <BarChartOutlined />,
          label: 'Repository Analysis',
          onClick: () => navigate('/analysis')
        },
        {
          key: 'commits',
          icon: <CodeOutlined />,
          label: 'Commits',
          onClick: () => navigate('/commits')
        },
        {
          key: 'team',
          icon: <TeamOutlined />,
          label: 'Team Analysis',
          onClick: () => navigate('/dashboard')
        }
      ]
    },
    {
      key: 'tools',
      icon: <RocketOutlined />,
      label: 'Công cụ',
      children: [
        {
          key: 'demo',
          icon: <ExperimentOutlined />,
          label: 'Component Demo',
          onClick: () => navigate('/demo')
        },
        {
          key: 'test',
          icon: <SettingOutlined />,
          label: 'Test Page',
          onClick: () => navigate('/test')
        }
      ]
    }
  ];

  const handleMenuClick = ({ key }) => {
    const item = findMenuItem(menuItems, key);
    if (item && item.onClick) {
      item.onClick();
    }
  };

  const findMenuItem = (items, key) => {
    for (const item of items) {
      if (item.key === key) return item;
      if (item.children) {
        const found = findMenuItem(item.children, key);
        if (found) return found;
      }
    }
    return null;
  };

  const sidebarStyle = {
    background: theme === 'dark' 
      ? 'linear-gradient(180deg, #1f2937 0%, #111827 100%)'
      : 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
    borderRight: theme === 'dark' 
      ? '1px solid #374151'
      : '1px solid #e2e8f0',
    boxShadow: '2px 0 8px rgba(0, 0, 0, 0.1)',
    position: 'fixed',
    left: 0,
    top: 0,
    bottom: 0,
    zIndex: 1000,
    height: '100vh',
    overflow: 'hidden'
  };

  const headerStyle = {
    padding: collapsed ? '16px 8px' : '16px 24px',
    borderBottom: theme === 'dark' 
      ? '1px solid #374151'
      : '1px solid #e2e8f0',
    background: 'transparent',
    display: 'flex',
    alignItems: 'center',
    justifyContent: collapsed ? 'center' : 'space-between',
    minHeight: '64px'
  };

  const footerStyle = {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: collapsed ? '12px 8px' : '12px 24px',
    borderTop: theme === 'dark' 
      ? '1px solid #374151'
      : '1px solid #e2e8f0',
    background: theme === 'dark'
      ? 'rgba(31, 41, 55, 0.8)'
      : 'rgba(248, 250, 252, 0.8)',
    backdropFilter: 'blur(10px)'
  };

  const logoStyle = {
    color: theme === 'dark' ? '#ffffff' : '#1f2937',
    fontSize: collapsed ? '18px' : '20px',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  };

  return (
    <Sider
      width={260}
      collapsedWidth={80}
      collapsed={collapsed}
      style={sidebarStyle}
      trigger={null}
    >
      {/* Header với Logo */}
      <div style={headerStyle}>
        <div style={logoStyle}>
          <RocketOutlined style={{ fontSize: collapsed ? '24px' : '28px' }} />
          {!collapsed && <span>MENU</span>}
        </div>
        {!collapsed && (
          <Button
            type="text"
            icon={<MenuFoldOutlined />}
            onClick={onToggle}
            style={{ 
              color: theme === 'dark' ? '#9ca3af' : '#6b7280',
              border: 'none'
            }}
          />
        )}
      </div>

      {/* Menu chính */}
      <div style={{ 
        height: 'calc(100vh - 160px)', 
        overflowY: 'auto',
        overflowX: 'hidden',
        padding: '16px 0'
      }}>
        <Menu
          theme={theme}
          mode="inline"
          selectedKeys={selectedKeys}
          items={menuItems}
          onClick={handleMenuClick}
          inlineIndent={collapsed ? 0 : 24}
          style={{ 
            border: 'none',
            background: 'transparent',
            fontSize: '14px'
          }}
        />
      </div>

      {/* Footer với User Info */}
      <div style={footerStyle}>
        {user ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Avatar 
              src={user.avatar_url} 
              icon={<UserOutlined />}
              size={collapsed ? 32 : 40}
            />
            {!collapsed && (
              <div style={{ flex: 1, minWidth: 0 }}>
                <Text 
                  strong 
                  style={{ 
                    color: theme === 'dark' ? '#ffffff' : '#1f2937',
                    fontSize: '14px',
                    display: 'block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {user.username}
                </Text>
                <Text 
                  type="secondary" 
                  style={{ 
                    color: theme === 'dark' ? '#ffffff' : '#1f2937',
                    fontSize: '12px',
                    display: 'block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {user.email}
                </Text>
              </div>
            )}
            <Tooltip title="Đăng xuất">
              <Button
                type="text"
                icon={<LogoutOutlined />}
                onClick={onLogout}
                size="small"
                style={{ 
                  color: theme === 'dark' ? '#ef4444' : '#dc2626',
                  border: 'none'
                }}
              />
            </Tooltip>
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <Button
              type="primary"
              icon={<UserOutlined />}
              onClick={() => navigate('/login')}
              size={collapsed ? 'small' : 'default'}
              style={{ width: '100%' }}
            >
              {!collapsed && 'Đăng nhập'}
            </Button>
          </div>
        )}
      </div>

      {/* Toggle button khi collapsed */}
      {collapsed && (
        <Button
          type="text"
          icon={<MenuUnfoldOutlined />}
          onClick={onToggle}
          style={{
            position: 'absolute',
            top: '16px',
            right: '-15px',
            width: '30px',
            height: '30px',
            borderRadius: '50%',
            background: theme === 'dark' ? '#374151' : '#ffffff',
            border: theme === 'dark' ? '1px solid #4b5563' : '1px solid #e2e8f0',
            color: theme === 'dark' ? '#ffffff' : '#1f2937',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            zIndex: 1001
          }}
        />
      )}
    </Sider>
  );
};

export default AppSidebar;
