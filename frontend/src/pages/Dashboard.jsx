import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, Avatar, Card, Grid, Space, Divider, Badge, message, Spin } from 'antd';
import { LogoutOutlined, GithubOutlined, NotificationOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import RepoList from '../components/repo/RepoList';
import OverviewCard from '../components/Dashboard/OverviewCard';
import AIInsightWidget from '../components/Dashboard/AIInsightWidget';
import ProjectTaskManager from '../components/Dashboard/ProjectTaskManager';
import RepoListFilter from '../components/Dashboard/RepoListFilter';
import TaskBoard from '../components/Dashboard/TaskBoard';
import ControlPanel from '../components/Dashboard/components/ControlPanel';
import SyncProgressNotification from '../components/common/SyncProgressNotification';
import axios from 'axios';
import CommitAnalyst from '../components/Dashboard/components/CommitAnalyst';
import RepoDiagnosisPanel from '../components/Dashboard/components/RepoDiagnosisPanel';

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

const MainLayout = styled.div`
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 24px;
  min-height: calc(100vh - 200px);

  @media (max-width: 1200px) {
    grid-template-columns: 250px 1fr;
    gap: 16px;
  }

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 16px;
  }
`;

const Sidebar = styled.div`
  position: sticky;
  top: 24px;
  height: fit-content;
  display: flex;
  flex-direction: column;
  gap: 16px;

  @media (max-width: 768px) {
    position: static;
    order: 2;
  }
`;

const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-width: 0; /* Để tránh overflow */
`;

const SidebarCard = styled(Card)`
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);

  .ant-card-body {
    padding: 16px;
  }

  .ant-card-head {
    padding: 12px 16px;
    border-bottom: 1px solid #f1f5f9;
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

  const [isSyncing, setIsSyncing] = useState(false);
  const [repositories, setRepositories] = useState([]);
  const [repoLoading, setRepoLoading] = useState(true);

  // AI Model States
  const [aiModel, setAiModel] = useState('multifusion'); // Default to multifusion
  const [useAI, setUseAI] = useState(true); // Whether to use AI analysis
  const [aiModelStatus, setAiModelStatus] = useState(null); // For HAN model status
  const [multiFusionV2Status, setMultiFusionV2Status] = useState(null); // For MultiFusion V2 model status

  // Commit Analysis States
  const [selectedRepoId, setSelectedRepoId] = useState(null); // Currently selected repository for analysis
  const [memberCommits, setMemberCommits] = useState(null); // Analysis for a specific member's commits
  const [allRepoCommitAnalysis, setAllRepoCommitAnalysis] = useState(null); // Analysis for all repo commits
  const [commitAnalysisLoading, setCommitAnalysisLoading] = useState(false);

  // Fetch AI model statuses
  useEffect(() => {
    const fetchModelStatuses = async () => {
      try {
        // Fetch HAN model status
        const hanStatusRes = await axios.get('http://localhost:8000/api/han-commit-analysis/1/model-status');
        setAiModelStatus(hanStatusRes.data);
        console.log("HAN Model Status:", hanStatusRes.data);

        // Fetch MultiFusion V2 model status
        const mfV2StatusRes = await axios.get('http://localhost:8000/api/multifusion-commit-analysis/1/ai/model-v2-status');
        setMultiFusionV2Status(mfV2StatusRes.data);
        console.log("MultiFusion V2 Model Status:", mfV2StatusRes.data);

        // Set default AI model based on availability
        if (mfV2StatusRes.data?.model_info?.is_available) {
          setAiModel('multifusion');
        } else if (hanStatusRes.data?.model_loaded) {
          setAiModel('han');
        } else {
          setUseAI(false); // Disable AI if no models are available
        }

      } catch (error) {
        console.error('Error fetching AI model statuses:', error);
        setUseAI(false); // Disable AI on error
      }
    };
    fetchModelStatuses();
  }, []);

  // Fetch commit analysis data based on selectedRepoId, aiModel, and useAI
  useEffect(() => {
    const fetchCommitAnalysis = async () => {
      if (!selectedRepoId || !useAI) {
        setMemberCommits(null);
        setAllRepoCommitAnalysis(null);
        return;
      }

      setCommitAnalysisLoading(true);
      const token = localStorage.getItem('access_token');
      if (!token) {
        message.error('Vui lòng đăng nhập lại!');
        setCommitAnalysisLoading(false);
        return;
      }

      try {
        let memberCommitsData = null;
        let allRepoCommitsData = null;
        const defaultMemberLogin = user?.username || 'octocat'; // Placeholder, replace with actual selected member

        if (aiModel === 'han') {
          // Fetch HAN analysis for a member
          const hanMemberRes = await axios.get(`http://localhost:8000/api/han-commit-analysis/${selectedRepoId}/members/${defaultMemberLogin}/commits-han`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          memberCommitsData = hanMemberRes.data.data; // Assuming data is nested under 'data'

        } else if (aiModel === 'multifusion') {
          // Fetch MultiFusion V2 analysis for a member
          const mfMemberRes = await axios.get(`http://localhost:8000/api/multifusion-commit-analysis/${selectedRepoId}/members/${defaultMemberLogin}/commits-v2`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          memberCommitsData = mfMemberRes.data; // Assuming data is directly the response

          // Fetch MultiFusion V2 analysis for all repo commits
          const mfAllRepoRes = await axios.get(`http://localhost:8000/api/multifusion-commit-analysis/${selectedRepoId}/commits/all/analysis`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          allRepoCommitsData = mfAllRepoRes.data.analysis; // Assuming data is nested under 'analysis'
        }

        setMemberCommits(memberCommitsData);
        setAllRepoCommitAnalysis(allRepoCommitsData);
        console.log("Fetched Member Commits:", memberCommitsData);
        console.log("Fetched All Repo Commits:", allRepoCommitsData);

      } catch (error) {
        console.error('Error fetching commit analysis:', error);
        message.error('Không thể tải dữ liệu phân tích commit!');
        setMemberCommits(null);
        setAllRepoCommitAnalysis(null);
      } finally {
        setCommitAnalysisLoading(false);
      }
    };

    fetchCommitAnalysis();
  }, [selectedRepoId, aiModel, useAI, user]); // user dependency for defaultMemberLogin

  // Pre-fetch repositories từ database ngầm trong nền
  const preloadRepositoriesFromDB = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      // Gọi API lấy repos từ database (không phải GitHub API)
      const response = await axios.get('http://localhost:8000/api/repositories', {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      setRepositories(response.data);
      console.log(`Pre-loaded ${response.data.length} repositories from database`);
      // Set the first repository as selected by default if available
      if (response.data.length > 0 && selectedRepoId === null) {
        setSelectedRepoId(response.data[0].id);
      }
    } catch (error) {
      console.error('Lỗi khi pre-load repositories:', error);
    } finally {
      setRepoLoading(false);
    }
  };

  // Pre-load repositories ngay khi vào Dashboard
  useEffect(() => {
    preloadRepositoriesFromDB();
  }, [selectedRepoId]); // Add selectedRepoId to dependencies to avoid re-setting it

  const syncAllRepositories = async () => {
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
  };  useEffect(() => {
    const storedProfile = localStorage.getItem('github_profile');
    if (!storedProfile) {
      navigate('/login');
    } else {
      setUser(JSON.parse(storedProfile));
      
      // Removed automatic sync - now only manual sync is allowed
      // Users must manually sync repositories using the sync buttons
    }  }, [navigate]);

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
      <HeaderCard variant="borderless">
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
          </UserInfoContainer>          <Space size={screens.md ? 16 : 8}>
            
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
          </Space>        </Space>
      </HeaderCard>

      {/* Main Layout với Sidebar và Content */}
      <MainLayout>        {/* Sidebar bên trái */}
        <Sidebar>
          {/* Overview Metrics trong Sidebar */}
          <OverviewCard sidebar={true} />
          
          {/* Quick Actions */}
          <SidebarCard 
            title={<SectionTitle level={5} style={{ fontSize: '14px' }}>Thao tác nhanh</SectionTitle>}
            size="small"
          >
            <Space direction="vertical" style={{ width: '100%' }} size="small">              <Button 
                type="default" 
                onClick={syncAllRepositories}
                loading={isSyncing}
                disabled={isSyncing}
                block
                size="small"
                style={{ backgroundColor: '#f8fafc', borderColor: '#e2e8f0' }}
              >
                {isSyncing ? 'Đang đồng bộ...' : 'Đồng bộ đầy đủ'}
              </Button>
            </Space>
          </SidebarCard>

          {/* Activity Summary */}
          <SidebarCard 
            title={<SectionTitle level={5} style={{ fontSize: '14px' }}>Hoạt động gần đây</SectionTitle>}
            size="small"
          >
            <Space direction="vertical" style={{ width: '100%' }} size="small">
              <div style={{ fontSize: '12px', color: '#666' }}>
                • Task "Trò game tăng độ khó" đã hoàn thành
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                • 2 repositories mới được đồng bộ
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                • AI phân tích 15 commits mới
              </div>
            </Space>
          </SidebarCard>
        </Sidebar>

        {/* Main Content bên phải */}
        <MainContent>          {/* Project Task Manager - Full Width */}
          <DashboardCard>
            <ProjectTaskManager 
              repositories={repositories}
              repoLoading={repoLoading}
            />
          </DashboardCard>

          {/* Repository Analysis */}
          <DashboardCard 
            title={
              <SectionTitle level={5}>
                <GithubOutlined />
                Repository Analysis
              </SectionTitle>
            }
          >
            <AIInsightWidget 
              aiModel={aiModel}
              useAI={useAI}
              aiModelStatus={aiModelStatus}
              multiFusionV2Status={multiFusionV2Status}
            />
            {/* RepoDiagnosisPanel: Manual diagnosis for selected repo */}
            <RepoDiagnosisPanel 
              repositories={repositories}
              selectedRepoId={selectedRepoId}
              onRepoChange={repo => setSelectedRepoId(repo?.id)}
            />
          </DashboardCard>

          {/* Filters Section */}
          <DashboardCard 
            title={<SectionTitle level={5}>Filters & Settings</SectionTitle>}
          >
            <RepoListFilter onFilterChange={handleFilterChange} />
          </DashboardCard>

          {/* Control Panel */}
          <DashboardCard
            title={<SectionTitle level={5}>Control Panel</SectionTitle>}
          >
            <ControlPanel
              branches={[]}
              selectedBranch={null}
              setSelectedBranch={() => {}}
              branchesLoading={false}
              aiModel={aiModel}
              setAiModel={setAiModel}
              useAI={useAI}
              setUseAI={setUseAI}
              aiModelStatus={aiModelStatus}
              multiFusionV2Status={multiFusionV2Status}
              showAIFeatures={false}
              setShowAIFeatures={() => {}}
              fullAnalysisLoading={false}
              onAnalyzeFullRepo={() => {}}
            />
          </DashboardCard>

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
            title={<SectionTitle level={5}>Commit Analysis</SectionTitle>}
          >
            <CommitAnalyst 
              memberCommits={memberCommits}
              allRepoCommitAnalysis={allRepoCommitAnalysis}
              loading={commitAnalysisLoading}
            />
          </DashboardCard>
          </ContentSection>
        </MainContent>
      </MainLayout>

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