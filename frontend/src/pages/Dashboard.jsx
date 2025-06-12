import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, Avatar, Card, Grid, Space, Divider, Badge, message, Spin } from 'antd';
import { LogoutOutlined, GithubOutlined, NotificationOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import RepoList from '../components/repo/RepoList';
import OverviewCard from '../components/Dashboard/OverviewCard';
import AIInsightWidget from '../components/Dashboard/AIInsightWidget';
import RepoListFilter from '../components/Dashboard/RepoListFilter';
import TaskBoard from '../components/Dashboard/TaskBoard';
import SyncProgressNotification from '../components/common/SyncProgressNotification';
import axios from 'axios';

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
  const [loading] = useState(false);
  const navigate = useNavigate();
  const screens = useBreakpoint();
  // Progress notification states
  const [syncProgress, setSyncProgress] = useState({
    visible: false,
    totalRepos: 0,
    completedRepos: 0,
    currentRepo: '',
    repoProgresses: [],
    overallProgress: 0
  });

  const [isSyncing, setIsSyncing] = useState(false);  const syncAllRepositories = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      message.error('Vui lòng đăng nhập lại!');
      return;
    }

    // Hiển thị progress ngay lập tức TRƯỚC khi set loading
    setSyncProgress({
      visible: true,
      totalRepos: 0,
      completedRepos: 0,
      currentRepo: 'Đang lấy danh sách repository...',
      repoProgresses: [],
      overallProgress: 0
    });

    setIsSyncing(true);

    try {
      // Thêm timeout nhỏ để đảm bảo UI render progress trước
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const response = await axios.get('http://localhost:8000/api/github/repos', {
        headers: {
          Authorization: `token ${token}`,
        },
      });

      const repositories = response.data;
      
      // Cập nhật với danh sách repository thực tế
      setSyncProgress(prev => ({
        ...prev,
        totalRepos: repositories.length,
        currentRepo: 'Chuẩn bị đồng bộ...',
        repoProgresses: repositories.map(repo => ({
          name: `${repo.owner.login}/${repo.name}`,
          status: 'pending',
          progress: 0
        }))
      }));

      let completedCount = 0;
      
      // Đồng bộ từng repository một cách tuần tự để tracking dễ hơn
      for (const repo of repositories) {
        const repoName = `${repo.owner.login}/${repo.name}`;
        
        // Cập nhật repository hiện tại
        setSyncProgress(prev => ({
          ...prev,
          currentRepo: repoName,
          repoProgresses: prev.repoProgresses.map(r => 
            r.name === repoName ? { ...r, status: 'syncing', progress: 0 } : r
          )
        }));

        try {
          // Đồng bộ repository
          await axios.post(
            `http://localhost:8000/api/github/${repo.owner.login}/${repo.name}/sync-all`,
            {},
            {
              headers: {
                Authorization: `token ${token}`,
              },
            }
          );

          completedCount++;
          
          // Cập nhật trạng thái hoàn thành
          setSyncProgress(prev => ({
            ...prev,
            completedRepos: completedCount,
            overallProgress: (completedCount / repositories.length) * 100,
            repoProgresses: prev.repoProgresses.map(r => 
              r.name === repoName ? { ...r, status: 'completed', progress: 100 } : r
            )
          }));

        } catch (error) {
          console.error(`Lỗi đồng bộ ${repoName}:`, error);
          
          // Cập nhật trạng thái lỗi
          setSyncProgress(prev => ({
            ...prev,
            repoProgresses: prev.repoProgresses.map(r => 
              r.name === repoName ? { ...r, status: 'error', progress: 0 } : r
            )
          }));
          
          completedCount++; // Vẫn tính là completed để tiếp tục
        }
      }

      message.success('Đồng bộ tất cả repository hoàn thành!');

    } catch (error) {
      console.error('Lỗi khi lấy danh sách repository:', error);
      message.error('Không thể lấy danh sách repository!');
      setSyncProgress(prev => ({ ...prev, visible: false }));
    } finally {
      setIsSyncing(false);
    }
  };
  useEffect(() => {
    const storedProfile = localStorage.getItem('github_profile');
    if (!storedProfile) {
      navigate('/login');
    } else {
      setUser(JSON.parse(storedProfile));
      
      // Đồng bộ cơ bản nhanh để hiển thị danh sách repo ngay lập tức
      syncBasicRepositories();
    }
  }, [navigate]);

  // Đồng bộ cơ bản (nhanh) - chỉ thông tin repo và branches
  const syncBasicRepositories = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const response = await axios.get('http://localhost:8000/api/github/repos', {
        headers: { Authorization: `token ${token}` },
      });

      const repositories = response.data;
      message.info(`Đồng bộ cơ bản ${repositories.length} repository...`);
      
      // Đồng bộ cơ bản song song (nhanh hơn)
      Promise.all(
        repositories.slice(0, 10).map(repo => // Chỉ đồng bộ 10 repo đầu tiên
          axios.post(
            `http://localhost:8000/api/github/${repo.owner.login}/${repo.name}/sync-basic`,
            {},
            { headers: { Authorization: `token ${token}` } }
          ).catch(() => null)
        )
      ).then(() => {
        message.success('Đồng bộ cơ bản hoàn thành!');
      });

    } catch (error) {
      console.error('Lỗi đồng bộ cơ bản:', error);
    }
  };

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

  if (loading) {
    return <Spin tip="Đang đồng bộ dữ liệu..." size="large" />;
  }

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
          </UserInfoContainer>            <Space size={screens.md ? 16 : 8}>
            <Button 
              type="default" 
              onClick={syncAllRepositories}
              loading={isSyncing}
              disabled={isSyncing}
              style={{ backgroundColor: '#f8fafc', borderColor: '#e2e8f0' }}
            >
              {isSyncing ? 'Đang đồng bộ...' : 'Đồng bộ đầy đủ'}
            </Button>
            
            {/* Test button for instant progress */}
            <Button 
              onClick={() => {
                setSyncProgress({
                  visible: true,
                  totalRepos: 5,
                  completedRepos: 0,
                  currentRepo: 'Test repository...',
                  repoProgresses: [],
                  overallProgress: 0
                });
              }}
              style={{ background: '#10b981', borderColor: '#10b981', color: 'white' }}
            >
              Test Progress
            </Button>
            
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
        </DashboardCard>        <DashboardCard 
          title={<SectionTitle level={5}>Project Tasks</SectionTitle>}
        >
          <TaskBoard onStatusChange={handleStatusChange} />
        </DashboardCard>
      </ContentSection>

      {/* Progress Notification */}
      <SyncProgressNotification
        visible={syncProgress.visible}
        onClose={() => setSyncProgress(prev => ({ ...prev, visible: false }))}
        totalRepos={syncProgress.totalRepos}
        completedRepos={syncProgress.completedRepos}
        currentRepo={syncProgress.currentRepo}
        repoProgresses={syncProgress.repoProgresses}
        overallProgress={syncProgress.overallProgress}
      />
    </DashboardContainer>
  );
};

export default Dashboard;