import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, Avatar, Card, Grid, Space, Divider, Badge } from 'antd';
import { LogoutOutlined, GithubOutlined, NotificationOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import RepoList from '../components/repo/RepoList';
import OverviewCard from '../components/Dashboard/OverviewCard';
import AIInsightWidget from '../components/Dashboard/AIInsightWidget';
import RepoListFilter from '../components/Dashboard/RepoListFilter';
import TaskBoard from '../components/Dashboard/TaskBoard';

const { Title, Text } = Typography;
const { useBreakpoint } = Grid;

// Styled components với theme hiện đại
const DashboardContainer = styled.div`
  padding: 24px;
  max-width: 1440px;
  margin: 0 auto;
  background: #f8fafc;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 24px;

  @media (max-width: 768px) {
    padding: 16px;
    gap: 16px;
  }
`;

const HeaderCard = styled(Card)`
  border-radius: 16px;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  
  .ant-card-body {
    padding: 24px;
  }
`;

const DashboardCard = styled(Card)`
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  transition: all 0.2s cubic-bezier(0.645, 0.045, 0.355, 1);
  
  &:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    transform: translateY(-2px);
  }

  .ant-card-head {
    border-bottom: 1px solid #f1f5f9;
    padding: 16px 24px;
  }

  .ant-card-body {
    padding: 24px;
  }

  @media (max-width: 768px) {
    .ant-card-body {
      padding: 16px;
    }
  }
`;

const PrimaryButton = styled(Button)`
  border-radius: 8px;
  font-weight: 500;
  height: 40px;
  padding: 0 20px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const UserInfoContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
`;

const UserAvatar = styled(Avatar)`
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  border: 2px solid #ffffff;
`;

const WidgetsRow = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 24px;

  @media (max-width: 992px) {
    grid-template-columns: 1fr;
  }
`;

const ContentSection = styled.section`
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const SectionTitle = styled(Title)`
  margin-bottom: 0 !important;
  font-weight: 600 !important;
  color: #1e293b !important;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const NotificationBadge = styled(Badge)`
  .ant-badge-count {
    background: #3b82f6;
    box-shadow: 0 0 0 1px #fff;
  }
`;

const Dashboard = () => {
  const [user, setUser] = useState(null);
  // Removed unused activeTab state
  const navigate = useNavigate();
  const screens = useBreakpoint();

  useEffect(() => {
    const storedProfile = localStorage.getItem('github_profile');
    if (!storedProfile) {
      navigate('/login');
    } else {
      setUser(JSON.parse(storedProfile));
    }
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('github_profile');
    localStorage.removeItem('access_token');
    navigate('/login');
  };

  const handleFilterChange = (filters) => {
    console.log('Applied filters:', filters);
  };

  const handleStatusChange = (taskId, newStatus) => {
    console.log(`Updated task ${taskId} status to ${newStatus}`);
  };

  return (
    <DashboardContainer>
      {/* Header Section */}
      <HeaderCard bordered={false}>
        <Space 
          direction={screens.md ? 'horizontal' : 'vertical'} 
          align={screens.md ? 'center' : 'start'}
          style={{ width: '100%', justifyContent: 'space-between' }}
        >
          <UserInfoContainer>
            <UserAvatar src={user?.avatar_url} size={screens.md ? 72 : 56} />
            <div>
              <Title level={4} style={{ margin: 0, color: '#1e293b' }}>
                Welcome back, {user?.username || 'User'}!
              </Title>
              <Text type="secondary" style={{ color: '#64748b' }}>
                {user?.email || 'No email provided'}
              </Text>
            </div>
          </UserInfoContainer>
          
          <Space size={screens.md ? 16 : 8}>
            <NotificationBadge count={3} size="small">
              <Button 
                icon={<NotificationOutlined />} 
                shape="circle" 
                style={{ border: 'none' }}
              />
            </NotificationBadge>
            <PrimaryButton 
              type="primary" 
              danger 
              onClick={handleLogout}
              icon={<LogoutOutlined />}
            >
              {screens.md ? 'Log Out' : ''}
            </PrimaryButton>
          </Space>
        </Space>
      </HeaderCard>

      {/* Overview Metrics */}
      <DashboardCard bodyStyle={{ padding: '16px' }}>
        <OverviewCard />
      </DashboardCard>

      {/* AI Insights and Filters */}
      <WidgetsRow>
        <DashboardCard 
          title={
            <SectionTitle level={5}>
              <GithubOutlined />
              Repository Analysis
            </SectionTitle>
          }
        >
          <AIInsightWidget />
        </DashboardCard>
        
        <DashboardCard 
          title={<SectionTitle level={5}>Filters & Settings</SectionTitle>}
        >
          <RepoListFilter onFilterChange={handleFilterChange} />
        </DashboardCard>
      </WidgetsRow>

      {/* Main Content Sections */}
      <ContentSection>
        <DashboardCard 
          title={
            <SectionTitle level={5}>
              My Repositories
              <Text type="secondary" style={{ fontSize: 14, marginLeft: 8 }}>
                (24 repositories)
              </Text>
            </SectionTitle>
          }
        >
          <RepoList />
        </DashboardCard>

        <DashboardCard 
          title={<SectionTitle level={5}>Project Tasks</SectionTitle>}
        >
          <TaskBoard onStatusChange={handleStatusChange} />
        </DashboardCard>
      </ContentSection>
    </DashboardContainer>
  );
};

export default Dashboard;