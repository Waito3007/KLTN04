// src/components/RepoList.jsx
import { useEffect, useState } from "react";
import { List, Avatar, Typography, Spin, message } from "antd";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { Button } from "antd";
import { GithubOutlined } from "@ant-design/icons";
const { Title } = Typography;

const RepoList = () => {
  const [repos, setRepos] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchRepos = async () => {
      const token = localStorage.getItem("access_token");
      if (!token) return message.error("Không có token!");

      try {
        const response = await axios.get("http://localhost:8000/api/github/repos", {
        headers: {
  Authorization: `token ${token}`,
},


        });
        setRepos(response.data);
      } catch (error) {
        message.error("Lỗi khi lấy danh sách repository!");
        console.error(error);
      } finally {
        setLoading(false);
      }
    };

    fetchRepos();
  }, []);

  if (loading) {
    return <Spin tip="Đang tải repository..." />;
  }

  return (
    <div style={{ marginTop: "2rem" }}>
      <Title level={3}>📂 Repository của bạn</Title>
      <List
  itemLayout="horizontal"
  dataSource={repos}
  renderItem={(repo) => (
    <List.Item
      style={{ cursor: "pointer" }}
      onClick={() => navigate(`/repo/${repo.owner.login}/${repo.name}`)}
    >
      <List.Item.Meta
        avatar={<Avatar src={repo.owner.avatar_url} />}
        title={<span>{repo.name}</span>}
        description={repo.description || "Không có mô tả"}
      />
    </List.Item>
  )}
/>
    </div>
  );
};

export default RepoList;
