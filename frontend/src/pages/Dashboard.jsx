import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import RepoList from "../components/repo/RepoList";
import { Button, Typography, Avatar, Space } from "antd";

const { Title, Text } = Typography;

function Dashboard() {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const storedProfile = localStorage.getItem("github_profile");
    if (!storedProfile) {
      navigate("/login");
    } else {
      setUser(JSON.parse(storedProfile));
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("github_profile");
    localStorage.removeItem("access_token");
    navigate("/login");
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Space align="center">
          <Avatar src={user?.avatar_url} size={80} />
          <div>
            <Title level={3}>👋 Xin chào, {user?.username}!</Title>
            <Text type="secondary">Email: {user?.email}</Text>
          </div>
        </Space>

        <Button onClick={handleLogout} type="primary" danger>
          Đăng xuất
        </Button>

        {/* GẮN DANH SÁCH REPO VÀO ĐÂY */}
        <RepoList />
      </Space>
    </div>
  );
}

export default Dashboard;
