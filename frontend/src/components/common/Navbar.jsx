import React from 'react';
import { Space, Badge, Typography, Avatar } from 'antd';
import { LogoutOutlined, NotificationOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';
import Button from "@components/common/Button";
import theme from './theme';

const { Title, Text } = Typography;

const HeaderCard = styled.div`
  border-radius: 16px;
  background: linear-gradient(135deg, ${theme.colors.white} 0%, ${theme.colors.light} 100%);
  border: 1px solid ${theme.colors.secondary};
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  padding: ${theme.spacing.lg};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const UserInfoContainer = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.md};
`;

const UserAvatar = styled(Avatar)`
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  border: 2px solid ${theme.colors.white};
`;

const PrimaryButton = styled(Button)`
  border-radius: 8px;
  font-weight: 500;
  height: 40px;
  padding: 0 ${theme.spacing.lg};
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
`;

const Navbar = ({ user, isSyncing, syncAllRepositories, handleLogout }) => {
  const navigate = useNavigate();

  return (
    <HeaderCard>
      <UserInfoContainer>
        <UserAvatar src={user?.avatar_url} size={72} />
        <div>
          <Title level={4} style={{ margin: 0, color: '#1e293b' }}>
            Welcome back, {user?.username || 'User'}!
          </Title>
          <Text type="secondary" style={{ color: '#64748b' }}>
            {user?.email || 'No email provided'}
          </Text>
        </div>
      </UserInfoContainer>
      <Space size={16}>
        <Button 
          type="default" 
          onClick={syncAllRepositories}
          loading={isSyncing}
          disabled={isSyncing}
          style={{ backgroundColor: '#f8fafc', borderColor: '#e2e8f0' }}
        >
          {isSyncing ? 'Đang đồng bộ...' : 'Đồng bộ đầy đủ'}
        </Button>
        <Button 
          type="primary" 
          onClick={() => navigate('/task-assignment')}
          style={{ backgroundColor: '#10b981', borderColor: '#10b981', color: 'white' }}
        >
          Quản lý nhiệm vụ
        </Button>
        <Button 
          type="primary" 
          onClick={() => navigate('/ai-insights')}
          style={{ backgroundColor: '#fa8c16', borderColor: '#faad14', color: 'white' }}
        >
          AI Insights
        </Button>
        <Badge count={3} size="small">
          <Button 
            icon={<NotificationOutlined />} 
            shape="circle" 
            style={{ border: 'none' }}
          />
        </Badge>
        <PrimaryButton 
          type="primary" 
          danger 
          onClick={handleLogout}
          icon={<LogoutOutlined />}
        >
          Log Out
        </PrimaryButton>
      </Space>
    </HeaderCard>
  );
};

export default Navbar;
