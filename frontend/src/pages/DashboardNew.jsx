import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Button, 
  Typography, 
  Avatar, 
  Card, 
  Row, 
  Col, 
  Space, 
  Divider, 
  Badge, 
  message, 
  Spin,
  Layout,
  Statistic
} from 'antd';
import { 
  LogoutOutlined, 
  GithubOutlined, 
  NotificationOutlined, 
  TeamOutlined, 
  ProjectOutlined,
  BarChartOutlined,
  RocketOutlined,
  BulbOutlined
} from '@ant-design/icons';
import RepoList from "@components/repo/RepoList";
import SyncProgressNotification from "@components/common/SyncProgressNotification";
import RepoDiagnosisPanel from "@components/Dashboard/components/RepoDiagnosisPanel";
import MemberSkillProfilePanel from "@components/Dashboard/MemberSkill/MemberSkillProfilePanel";
import DashboardAnalyst from "@components/Dashboard/Dashboard_Analyst/DashboardAnalyst";
import TaskAssignBoard from "@components/Dashboard/TaskAssign/TaskAssignBoard";
import MainLayout from "@components/layout/MainLayout";
import { theme } from "@components/common";

const { Title, Text } = Typography;
const { Header, Content } = Layout;

const Dashboard = () => {
  const navigate = useNavigate();
  
  // State management
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState({
    username: 'Team Leader',
    avatar_url: '',
    email: 'leader@company.com'
  });

  // Stats state
  const [stats, setStats] = useState({
    totalRepos: 0,
    activeProjects: 0,
    completedTasks: 0,
    teamMembers: 0
  });

  // Fetch data on component mount
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        // Simulate API calls
        setTimeout(() => {
          setRepositories([
            { id: 1, name: 'project-alpha', status: 'active', commits: 42 },
            { id: 2, name: 'project-beta', status: 'completed', commits: 28 },
            { id: 3, name: 'project-gamma', status: 'planning', commits: 5 }
          ]);
          setStats({
            totalRepos: 12,
            activeProjects: 5,
            completedTasks: 84,
            teamMembers: 8
          });
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const handleLogout = () => {
    // Implement logout logic
    navigate('/login');
  };

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        background: theme.colors.bg.tertiary
      }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <MainLayout variant="modern" padding={0}>
      <Layout style={{ minHeight: '100vh' }}>
        {/* Dashboard Header */}
        <Header style={{
          background: 'white',
          padding: '0 24px',
          boxShadow: theme.shadows.small,
          borderBottom: `1px solid ${theme.colors.light}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <RocketOutlined style={{ fontSize: '24px', color: theme.colors.primary }} />
            <Title level={3} style={{ margin: 0, color: theme.colors.dark }}>
              TaskFlowAI Dashboard
            </Title>
          </div>
          
          <Space size="large">
            <Badge count={5} size="small">
              <Button type="text" icon={<NotificationOutlined />} />
            </Badge>
            <Divider type="vertical" />
            <Space>
              <Avatar 
                src={user.avatar_url} 
                icon={<TeamOutlined />}
                style={{ border: `2px solid ${theme.colors.primary}` }}
              />
              <div>
                <Text strong>{user.username}</Text>
                <br />
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {user.email}
                </Text>
              </div>
              <Button 
                type="text" 
                icon={<LogoutOutlined />} 
                onClick={handleLogout}
                danger
              >
                Đăng xuất
              </Button>
            </Space>
          </Space>
        </Header>

        <Content style={{ padding: '24px', background: theme.colors.bg.tertiary }}>
          {/* Quick Stats */}
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24} sm={12} lg={6}>
              <Card style={{ borderRadius: theme.borderRadius.large }}>
                <Statistic
                  title="Tổng Repository"
                  value={stats.totalRepos}
                  prefix={<ProjectOutlined style={{ color: theme.colors.primary }} />}
                  valueStyle={{ color: theme.colors.primary }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} lg={6}>
              <Card style={{ borderRadius: theme.borderRadius.large }}>
                <Statistic
                  title="Dự án đang hoạt động"
                  value={stats.activeProjects}
                  prefix={<RocketOutlined style={{ color: theme.colors.techGreen }} />}
                  valueStyle={{ color: theme.colors.techGreen }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} lg={6}>
              <Card style={{ borderRadius: theme.borderRadius.large }}>
                <Statistic
                  title="Task hoàn thành"
                  value={stats.completedTasks}
                  prefix={<BarChartOutlined style={{ color: theme.colors.success }} />}
                  valueStyle={{ color: theme.colors.success }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} lg={6}>
              <Card style={{ borderRadius: theme.borderRadius.large }}>
                <Statistic
                  title="Thành viên team"
                  value={stats.teamMembers}
                  prefix={<TeamOutlined style={{ color: theme.colors.techPurple }} />}
                  valueStyle={{ color: theme.colors.techPurple }}
                />
              </Card>
            </Col>
          </Row>

          {/* Main Dashboard Content */}
          <Row gutter={[24, 24]}>
            {/* Left Column */}
            <Col xs={24} lg={16}>
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                {/* Repository List */}
                <Card 
                  title={
                    <Space>
                      <GithubOutlined />
                      <span>Repositories</span>
                    </Space>
                  }
                  extra={
                    <Button type="primary" onClick={() => navigate('/sync')}>
                      Sync Repositories
                    </Button>
                  }
                  style={{ borderRadius: theme.borderRadius.large }}
                >
                  <RepoList />
                </Card>

                {/* Task Assignment Board */}
                <Card 
                  title={
                    <Space>
                      <BulbOutlined />
                      <span>Task Management</span>
                    </Space>
                  }
                  style={{ borderRadius: theme.borderRadius.large }}
                >
                  <TaskAssignBoard />
                </Card>
              </Space>
            </Col>

            {/* Right Column */}
            <Col xs={24} lg={8}>
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                {/* Sync Progress */}
                <Card 
                  title="Sync Status"
                  style={{ borderRadius: theme.borderRadius.large }}
                >
                  <SyncProgressNotification />
                </Card>

                {/* Repository Diagnosis */}
                <Card 
                  title="Phân tích Repository"
                  style={{ borderRadius: theme.borderRadius.large }}
                >
                  <RepoDiagnosisPanel />
                </Card>

                {/* Member Skill Profile */}
                <Card 
                  title="Team Skills"
                  style={{ borderRadius: theme.borderRadius.large }}
                >
                  <MemberSkillProfilePanel />
                </Card>

                {/* Dashboard Analytics */}
                <Card 
                  title="Analytics"
                  style={{ borderRadius: theme.borderRadius.large }}
                >
                  <DashboardAnalyst />
                </Card>
              </Space>
            </Col>
          </Row>
        </Content>
      </Layout>
    </MainLayout>
  );
};

export default Dashboard;
