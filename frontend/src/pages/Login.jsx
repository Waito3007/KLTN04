// src/pages/Login.jsx
import React from "react";
import { Button, Card, Typography } from "antd";
import { GithubOutlined } from "@ant-design/icons";

const { Title } = Typography;

const Login = () => {
  const handleGitHubLogin = () => {
    window.location.href = "http://localhost:8000/api/login"; // backend redirect to GitHub OAuth
  };

  return (
    <div className="h-screen flex items-center justify-center bg-gradient-to-br from-gray-100 to-white">
      <Card
        className="shadow-xl rounded-2xl w-full max-w-md"
        style={{ textAlign: "center", padding: "3rem 2rem" }}
      >
        <Title level={2} style={{ marginBottom: "2rem" }}>
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
          }}
        >
          Đăng nhập với GitHub
        </Button>
      </Card>
    </div>
  );
};

export default Login;