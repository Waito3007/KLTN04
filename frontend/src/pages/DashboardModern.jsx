import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Button, 
  Typography, 
  Avatar, 
  Row, 
  Col, 
  Space, 
  Divider, 
  Badge, 
  Layout,
  Statistic,
  Grid
} from 'antd';
import { 
  LogoutOutlined, 
  GithubOutlined, 
  NotificationOutlined, 
  TeamOutlined, 
  ProjectOutlined,
  BarChartOutlined,
  RocketOutlined,
  BulbOutlined,
  UserOutlined,
  CloudOutlined,
  DashboardOutlined
} from '@ant-design/icons';
import SyncProgressNotification from "@components/common/SyncProgressNotification";
import { Loading, Toast, Card, Modal, EmptyState } from "@components/common";
import axios from 'axios';
import MemberSkillProfilePanel from "@components/Dashboard/MemberSkill/MemberSkillProfilePanel";
import DashboardAnalyst from "@components/Dashboard/Dashboard_Analyst/DashboardAnalyst";
import TaskAssignBoard from "@components/Dashboard/TaskAssign/TaskAssignBoard";

const { Title, Text } = Typography;

// Styles object cho modern design
const styles = {
  container: {
    minHeight: 'auto', // MainLayout sẽ handle background
  },
  
  innerContainer: {
    background: 'rgba(255, 255, 255, 0.98)',
    borderRadius: '20px',
    boxShadow: '0 24px 48px rgba(0, 0, 0, 0.08), 0 8px 16px rgba(0, 0, 0, 0.04)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    padding: '32px',
    margin: '20px 0',
    position: 'relative',
    overflow: 'hidden'
  },

  header: {
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%)',
    borderRadius: '16px',
    border: '1px solid rgba(226, 232, 240, 0.3)',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.06)',
    marginBottom: '32px',
    padding: '28px',
    position: 'relative',
    overflow: 'hidden'
  },

  userInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    marginBottom: '16px',
  },

  avatar: {
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    border: '3px solid #ffffff',
  },

  logoutButton: {
    borderRadius: '12px',
    height: '40px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    border: 'none',
    color: 'white',
    fontWeight: '500',
    boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },

  mainGrid: {
    display: 'grid',
    gridTemplateColumns: '280px 1fr',
    gap: '24px',
    minHeight: 'calc(100vh - 300px)',
  },

  sidebar: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },

  sidebarCard: {
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
    background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  },

  mainContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
    minWidth: 0,
  },

  dashboardCard: {
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
    transition: 'all 0.3s cubic-bezier(0.645, 0.045, 0.355, 1)',
    background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  },

  sectionTitle: {
    margin: '0 0 16px 0',
    fontWeight: '600',
    color: '#1e293b',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },

  statsCard: {
    borderRadius: '12px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    border: 'none',
    color: 'white',
    textAlign: 'center',
  },

  responsiveGrid: {
    '@media (max-width: 1200px)': {
      gridTemplateColumns: '250px 1fr',
      gap: '16px',
    },
    '@media (max-width: 768px)': {
      gridTemplateColumns: '1fr',
      gap: '16px',
    },
  },
};

// Hàm buildApiUrl để xây dựng URL API
const buildApiUrl = (endpoint) => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
  return `${baseUrl}${endpoint}`;
};

const DashboardModern = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true); // Bắt đầu với loading = true
  const navigate = useNavigate();
  
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
  const [aiModel] = useState('multifusion');
  const [useAI] = useState(true);

  // Commit Analysis States
  const [selectedRepoId, setSelectedRepoId] = useState(null);
  // const [selectedBranch, setSelectedBranch] = useState(''); // Hidden - not used when MemberSkillProfilePanel is commented

  // Pre-fetch repositories từ database
  const preloadRepositoriesFromDB = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const response = await axios.get(buildApiUrl('/repositories'), {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      setRepositories(response.data);
      console.log(`Pre-loaded ${response.data.length} repositories from database`);
    } catch (error) {
      console.error('Lỗi khi pre-load repositories:', error);
    } finally {
      setRepoLoading(false);
    }
  };

  // Pre-load repositories effect
  useEffect(() => {
    if (repositories.length === 0) {
      preloadRepositoriesFromDB();
    }
  }, [repositories.length]);

  const syncAllRepositories = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      Toast.error('Vui lòng đăng nhập lại!');
      return;
    }

    setIsSyncing(true);
    setSyncProgress({
      visible: true,
      totalRepos: 0,
      completedRepos: 0,
      currentRepo: 'Đang khởi tạo...',
      repoProgresses: [],
      overallProgress: 0
    });

    try {
      Toast.info('Đang đồng bộ repositories từ GitHub...');
      
      const response = await axios.post(buildApiUrl('/repositories/sync-all'), {}, {
        headers: { 'Authorization': `Bearer ${token}` },
        timeout: 300000, // 5 minutes timeout
      });

      if (response.data?.status === 'success') {
        Toast.success(`Đồng bộ thành công! Đã đồng bộ ${response.data.synced_count} repositories.`);
        
        // Update progress to show completion
        setSyncProgress(prev => ({
          ...prev,
          totalRepos: response.data.total_repos,
          completedRepos: response.data.synced_count,
          currentRepo: 'Hoàn thành',
          overallProgress: 100
        }));
        
        // Reload repositories data
        await preloadRepositoriesFromDB();
        
      } else if (response.data?.status === 'partial_success') {
        Toast.warning(`Đồng bộ một phần! Đã đồng bộ ${response.data.synced_count}/${response.data.total_repos} repositories.`);
        
        // Update progress to show partial completion
        setSyncProgress(prev => ({
          ...prev,
          totalRepos: response.data.total_repos,
          completedRepos: response.data.synced_count,
          currentRepo: 'Hoàn thành một phần',
          overallProgress: (response.data.synced_count / response.data.total_repos) * 100
        }));
        
        // Show errors if any
        if (response.data.errors && response.data.errors.length > 0) {
          console.error('Sync errors:', response.data.errors);
          Toast.error(`Có ${response.data.errors.length} lỗi trong quá trình đồng bộ. Kiểm tra console để biết chi tiết.`);
        }
        
        await preloadRepositoriesFromDB();
        
      } else {
        Toast.error(response.data?.message || 'Có lỗi xảy ra khi đồng bộ!');
        
        setSyncProgress(prev => ({
          ...prev,
          currentRepo: 'Lỗi đồng bộ',
          overallProgress: 0
        }));
      }
      
    } catch (error) {
      console.error('Error syncing repositories:', error);
      
      let errorMessage = 'Có lỗi xảy ra khi đồng bộ repositories!';
      
      if (error.response?.status === 401) {
        errorMessage = 'Phiên đăng nhập hết hạn! Vui lòng đăng nhập lại.';
        Toast.error(errorMessage);
        navigate('/login');
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Đồng bộ mất quá nhiều thời gian! Vui lòng thử lại sau.';
        Toast.error(errorMessage);
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
        Toast.error(errorMessage);
      } else {
        errorMessage = error.message || errorMessage;
        Toast.error(errorMessage);
      }
      
      setSyncProgress(prev => ({
        ...prev,
        currentRepo: `Lỗi: ${errorMessage}`,
        overallProgress: 0
      }));
      
    } finally {
      setIsSyncing(false);
      
      // Hide progress after 3 seconds
      setTimeout(() => {
        setSyncProgress(prev => ({ ...prev, visible: false }));
      }, 3000);
    }
  };

  // Get user info from localStorage
  useEffect(() => {
    // Kiểm tra cả 'user' và 'github_profile' để tương thích
    const userData = localStorage.getItem('user') || localStorage.getItem('github_profile');
    const token = localStorage.getItem('access_token');
    
    console.log('🔍 Dashboard Auth Check:', {
      userData: !!userData,
      token: !!token,
      userDataContent: userData ? JSON.parse(userData) : null
    });
    
    if (userData && token) {
      setUser(JSON.parse(userData));
      setLoading(false); // Dừng loading khi có dữ liệu
    } else {
      // Chỉ redirect đến login nếu thực sự không có dữ liệu xác thực
      console.log('❌ No auth data found, redirecting to login');
      setLoading(false);
      navigate('/login');
    }
  }, [navigate]);

  if (loading) {
    return (
      <Loading 
        variant="gradient"
        text="Đang tải dashboard..."
        size="large"
      />
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.innerContainer}>
        {/* Header Section với thông tin user */}
        <div style={styles.header}>
          <div style={styles.userInfo}>
            <Avatar 
              size={64} 
              src={user?.avatar_url} 
              icon={<UserOutlined />}
              style={styles.avatar}
            />
            <div style={{ flex: 1 }}>
              <Title level={3} style={{ margin: 0, color: '#1e293b' }}>
                Chào mừng trở lại, {user?.name || user?.username || 'Developer'}! 🚀
              </Title>
              <Text type="secondary" style={{ fontSize: '16px' }}>
                <GithubOutlined /> {user?.username} • Dashboard AI Project Management
              </Text>
            </div>
            {/* <Button 
              type="primary" 
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              style={styles.logoutButton}
            >
              Đăng xuất
            </Button> */}
          </div>

          {/* Stats Overview */}
          <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
            <Col xs={24} sm={12} md={6}>
              <Card variant="stats">
                <Statistic 
                  title={<span style={{ color: 'rgba(255,255,255,0.8)' }}>Repositories</span>}
                  value={repositories.length} 
                  prefix={<GithubOutlined />}
                  valueStyle={{ color: 'white' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card variant="stats">
                <Statistic 
                  title={<span style={{ color: 'rgba(255,255,255,0.8)' }}>AI Model</span>}
                  value={aiModel === 'multifusion' ? 'MultiFusion V2' : 'HAN'} 
                  prefix={<RocketOutlined />}
                  valueStyle={{ color: 'white', fontSize: '16px' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card variant="stats">
                <Statistic 
                  title={<span style={{ color: 'rgba(255,255,255,0.8)' }}>Status</span>}
                  value={useAI ? 'Active' : 'Offline'} 
                  prefix={<CloudOutlined />}
                  valueStyle={{ color: 'white', fontSize: '16px' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card variant="stats">
                <Statistic 
                  title={<span style={{ color: 'rgba(255,255,255,0.8)' }}>Selected Repo</span>}
                  value={selectedRepoId ? 'Selected' : 'None'} 
                  prefix={<DashboardOutlined />}
                  valueStyle={{ color: 'white', fontSize: '16px' }}
                />
              </Card>
            </Col>
          </Row>
        </div>

        {/* Main Content Grid */}
        <div style={styles.mainGrid}>
          {/* Sidebar */}
          <div style={styles.sidebar}>
            {/* Quick Actions */}
            <Card 
              variant="modern"
              title={
                <Title level={5} style={styles.sectionTitle}>
                  <BulbOutlined /> Thao tác nhanh
                </Title>
              }
            >
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Button 
                  type="primary"
                  onClick={() => navigate('/repo-sync')}
                  block
                  style={styles.logoutButton}
                >
                  Repository Sync Manager
                </Button>
                
                <Button 
                  type="default" 
                  onClick={syncAllRepositories}
                  loading={isSyncing}
                  disabled={isSyncing}
                  block
                  style={{
                    borderRadius: '8px',
                    height: '40px',
                    fontWeight: '500',
                    ...(isSyncing && {
                      backgroundColor: '#f6ffed',
                      borderColor: '#b7eb8f',
                      color: '#389e0d'
                    })
                  }}
                >
                  {isSyncing ? 'Đang đồng bộ từ GitHub...' : 'Đồng bộ tất cả từ GitHub'}
                </Button>
                
                {/* Progress info when syncing */}
                {isSyncing && syncProgress.visible && (
                  <div style={{
                    padding: '8px 12px',
                    backgroundColor: '#f6ffed',
                    border: '1px solid #b7eb8f',
                    borderRadius: '6px',
                    fontSize: '12px'
                  }}>
                    <div style={{ color: '#389e0d', fontWeight: '500' }}>
                      {syncProgress.totalRepos > 0 ? 
                        `${syncProgress.completedRepos}/${syncProgress.totalRepos} repositories` :
                        'Đang khởi tạo...'
                      }
                    </div>
                    {syncProgress.currentRepo && (
                      <div style={{ color: '#666', marginTop: '2px' }}>
                        {syncProgress.currentRepo}
                      </div>
                    )}
                  </div>
                )}
              </Space>
            </Card>

            {/* Dashboard Analyst */}
            <Card 
              variant="modern"
              title={
                <Title level={5} style={styles.sectionTitle}>
                  <BarChartOutlined /> AI Analysis
                </Title>
              }
            >
              <DashboardAnalyst 
                selectedRepoId={selectedRepoId}
                repositories={repositories}
                // onBranchChange={setSelectedBranch} // Hidden - setSelectedBranch commented out
              />
            </Card>
          </div>

          {/* Main Content */}
          <div style={styles.mainContent}>
            {/* Task Assignment Board */}
            <Card 
              variant="modern"
              title={
                <Title level={4} style={styles.sectionTitle}>
                  <ProjectOutlined /> Task Assignment & Management
                </Title>
              }
            >
              <TaskAssignBoard 
                repositories={repositories}
                repoLoading={repoLoading}
                selectedRepoId={selectedRepoId}
                onRepoChange={(repo) => {
                  console.log('📥 Dashboard: Received repo change:', repo);
                  const newRepoId = repo?.id || null;
                  console.log('🔄 Dashboard: Setting selectedRepoId to:', newRepoId);
                  setSelectedRepoId(newRepoId);
                }}
              />
            </Card>

            {/* Member Skill Profiles - HIDDEN */}
            {/* <Card 
              variant="modern"
              title={
                <Title level={4} style={styles.sectionTitle}>
                  <TeamOutlined /> Member Skill Profiles
                </Title>
              }
            >
              <MemberSkillProfilePanel 
                repositories={repositories}
                selectedRepoId={selectedRepoId}
                selectedBranch={selectedBranch}
              />
            </Card> */}
          </div>
        </div>

        {/* Progress Notification */}
        <SyncProgressNotification
          visible={syncProgress.visible}
          onClose={() => setSyncProgress(prev => ({ ...prev, visible: false }))}
          totalRepos={syncProgress.totalRepos}
          completedRepos={syncProgress.completedRepos}
          currentRepo={syncProgress.currentRepo}
          repoProgresses={syncProgress.repoProgresses}
          overallProgress={syncProgress.overallProgress}
          title={isSyncing ? "Đồng bộ tất cả repositories từ GitHub" : "Đồng bộ hoàn thành"}
        />
      </div>
    </div>
  );
};

export default DashboardModern;
