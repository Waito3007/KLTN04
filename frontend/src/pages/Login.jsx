// src/pages/Login.jsx
import React from "react";
import { Button, Typography, Row, Col, Space, Divider } from "antd";
import { 
  GithubOutlined, 
  RobotOutlined, 
  TeamOutlined, 
  BarChartOutlined,
  BulbOutlined,
  RocketOutlined,
  CodeOutlined,
  BranchesOutlined
} from "@ant-design/icons";
import { buildApiUrl } from "@config/api";
import { theme, Card } from "@components/common";
import MainLayout from '@components/layout/MainLayout';
import "@styles/LoginAnimations.css";

const { Title, Paragraph, Text } = Typography;

const Login = () => {
  console.log("Login component is rendering...");
  
  const handleGitHubLogin = () => {
    console.log("GitHub login button clicked");
    window.location.href = buildApiUrl("/login");
  };

  const features = [
    {
      icon: <RobotOutlined style={{ fontSize: '24px', color: theme.colors.techPurple }} />,
      title: "AI Thông Minh",
      description: "Phân tích tiến độ và đề xuất công việc tự động"
    },
    {
      icon: <TeamOutlined style={{ fontSize: '24px', color: theme.colors.techBlue }} />,
      title: "Quản Lý Nhóm",
      description: "Phân công công việc hiệu quả cho từng thành viên"
    },
    {
      icon: <BarChartOutlined style={{ fontSize: '24px', color: theme.colors.techGreen }} />,
      title: "Báo Cáo Realtime",
      description: "Theo dõi tiến độ và hiệu suất theo thời gian thực"
    },
    {
      icon: <BulbOutlined style={{ fontSize: '24px', color: theme.colors.warning }} />,
      title: "Đề Xuất Thông Minh",
      description: "Gợi ý tối ưu hóa quy trình làm việc"
    }
  ];

  console.log("Login component rendered successfully");
  
  return (
    <MainLayout variant="gradient" padding={0}>
      <div style={{
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden'
      }}>
      {/* Background decorative elements */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          radial-gradient(circle at 20% 20%, rgba(114, 46, 209, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 80% 80%, rgba(24, 144, 255, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 40% 60%, rgba(0, 185, 107, 0.1) 0%, transparent 50%)
        `,
        zIndex: 1
      }} />
      
      {/* Floating icons */}
      <div style={{
        position: 'absolute',
        top: '10%',
        left: '10%',
        fontSize: '48px',
        color: 'rgba(255, 255, 255, 0.1)',
        animation: 'float 6s ease-in-out infinite',
        zIndex: 1
      }}>
        <CodeOutlined />
      </div>
      <div style={{
        position: 'absolute',
        top: '20%',
        right: '15%',
        fontSize: '36px',
        color: 'rgba(255, 255, 255, 0.1)',
        animation: 'float 4s ease-in-out infinite 2s',
        zIndex: 1
      }}>
        <BranchesOutlined />
      </div>
      <div style={{
        position: 'absolute',
        bottom: '20%',
        left: '15%',
        fontSize: '42px',
        color: 'rgba(255, 255, 255, 0.1)',
        animation: 'float 5s ease-in-out infinite 1s',
        zIndex: 1
      }}>
        <RocketOutlined />
      </div>

      <div style={{
        position: 'relative',
        zIndex: 2,
        padding: '40px 20px',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Row gutter={[48, 48]} align="middle" style={{ maxWidth: '1200px', width: '100%' }}>
          {/* Left side - Hero content */}
          <Col xs={24} lg={12}>
            <div style={{ textAlign: 'left', color: 'white' }} className="animate-slideInLeft">
              <Title level={1} style={{ 
                color: 'white', 
                fontSize: '3.5rem', 
                fontWeight: 700,
                marginBottom: '24px',
                lineHeight: 1.2
              }} className="hero-title">
                TaskFlow<span style={{ color: theme.colors.warning }}>AI</span>
              </Title>
              
              <Title level={2} style={{ 
                color: 'rgba(255, 255, 255, 0.9)', 
                fontSize: '1.5rem',
                fontWeight: 400,
                marginBottom: '32px'
              }} className="hero-subtitle">
                Ứng dụng AI hỗ trợ quản lý tiến độ và phân công công việc trong dự án lập trình
              </Title>

              <Paragraph style={{ 
                fontSize: '1.1rem', 
                color: 'rgba(255, 255, 255, 0.8)',
                marginBottom: '40px',
                lineHeight: 1.6
              }}>
                Tối ưu hóa quy trình làm việc của team với sức mạnh AI. 
                Phân tích thông minh, đề xuất chính xác, quản lý hiệu quả.
              </Paragraph>

              {/* Features grid */}
              <Row gutter={[24, 24]} style={{ marginBottom: '40px' }}>
                {features.map((feature, index) => (
                  <Col xs={12} sm={12} key={index}>
                    <Card
                      variant="glassMorphism"
                      size="small"
                      className="feature-card animate-fadeInUp"
                      hoverable={true}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-5px)';
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                      }}
                    >
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        {feature.icon}
                        <Text strong style={{ color: 'white', fontSize: '14px' }}>
                          {feature.title}
                        </Text>
                        <Text style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '12px' }}>
                          {feature.description}
                        </Text>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </div>
          </Col>

          {/* Right side - Login card */}
          <Col xs={24} lg={12}>
            <div style={{ display: 'flex', justifyContent: 'center' }} className="animate-slideInRight">
              <Card
                variant="modern"
                style={{ 
                  width: '100%',
                  maxWidth: '450px',
                  background: 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(20px)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-5px)';
                  e.currentTarget.style.boxShadow = theme.shadows.large + ', 0 20px 40px rgba(0,0,0,0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = theme.shadows.large;
                }}
              >
                <div style={{ textAlign: 'center' }}>
                  <div style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    background: theme.colors.gradient.tech,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 24px',
                    boxShadow: theme.shadows.glow
                  }} className="animate-pulse">
                    <RobotOutlined style={{ fontSize: '36px', color: 'white' }} />
                  </div>

                  <Title level={2} style={{ 
                    marginBottom: '16px', 
                    color: theme.colors.dark,
                    fontSize: '1.8rem'
                  }}>
                    Chào mừng đến với TaskFlowAI
                  </Title>
                  
                  <Paragraph style={{ 
                    color: theme.colors.secondary, 
                    marginBottom: '32px',
                    fontSize: '16px'
                  }}>
                    Đăng nhập để khám phá sức mạnh của AI trong quản lý dự án
                  </Paragraph>

                  <Button
                    type="primary"
                    icon={<GithubOutlined />}
                    size="large"
                    onClick={handleGitHubLogin}
                    style={{
                      width: "100%",
                      height: '56px',
                      fontSize: '16px',
                      fontWeight: 600,
                      borderRadius: theme.borderRadius.medium,
                      background: theme.colors.dark,
                      borderColor: theme.colors.dark,
                      boxShadow: theme.shadows.medium,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      transition: 'all 0.3s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = theme.shadows.large;
                      e.currentTarget.style.background = theme.colors.gradient.tech;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = theme.shadows.medium;
                      e.currentTarget.style.background = theme.colors.dark;
                    }}
                  >
                    Đăng nhập với GitHub
                  </Button>

                  <Divider style={{ margin: '32px 0 24px' }}>
                    <Text style={{ color: theme.colors.secondary, fontSize: '14px' }}>
                      Hoặc tìm hiểu thêm
                    </Text>
                  </Divider>

                  <div style={{ 
                    textAlign: 'center',
                    color: theme.colors.secondary,
                    fontSize: '14px',
                    lineHeight: 1.6
                  }}>
                    <p>✨ Phân tích commit và pull request tự động</p>
                    <p>🚀 Đề xuất task và phân công thông minh</p>
                    <p>📊 Dashboard analytics chi tiết</p>
                    <p>🔔 Cảnh báo và báo cáo realtime</p>
                  </div>
                </div>
              </Card>
            </div>
          </Col>
        </Row>
      </div>
      </div>
    </MainLayout>
  );
};

export default Login;