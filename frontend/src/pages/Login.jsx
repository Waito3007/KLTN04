// src/pages/Login.jsx
import React from "react";
import { Button, Card, Typography } from "antd";
import { GithubOutlined } from "@ant-design/icons";
import { buildApiUrl } from "../config/api";

const { Title } = Typography;

const Login = () => {
  console.log("Login component is rendering...");
  
  const handleGitHubLogin = () => {
    console.log("GitHub login button clicked");
    window.location.href = buildApiUrl("/login"); // backend redirect to GitHub OAuth
  };

  console.log("Login component rendered successfully");
  
  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      padding: '20px'
    }}>
      <Card
        style={{ 
          textAlign: "center", 
          padding: "3rem 2rem",
          width: '100%',
          maxWidth: '400px',
          borderRadius: '16px',
          boxShadow: '0 10px 25px rgba(0,0,0,0.1)'
        }}
      >
        <Title level={2} style={{ marginBottom: "2rem", color: '#2c3e50' }}>
          Đăng nhập vào <span style={{ color: "#1890ff" }}>TaskFlowAI</span>
        </Title>
        
        <Button
          type="primary"
          icon={<GithubOutlined />}
          size="large"
          onClick={handleGitHubLogin}
          style={{
            backgroundColor: "#000",
            borderColor: "#000",
            width: "100%",
            height: '48px',
            fontSize: '16px',
            borderRadius: '8px'
          }}
        >
          Đăng nhập với GitHub
        </Button>
        
        <div style={{ marginTop: '24px', color: '#6c757d', fontSize: '14px' }}>
          <p>Sử dụng tài khoản GitHub để đăng nhập</p>
          <p>Hệ thống quản lý task và phân tích commits</p>
        </div>
      </Card>
    </div>
  );
};

export default Login;