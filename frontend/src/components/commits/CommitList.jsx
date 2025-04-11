import { useEffect, useState } from "react";
import { List, Avatar, Typography, Spin, message } from "antd";
import axios from "axios";

const { Title } = Typography;

const CommitList = ({ owner, repo, branch }) => {
  const [commits, setCommits] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!branch) return;

    const token = localStorage.getItem("access_token");
    if (!token) return;

    const fetchCommits = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8000/api/github/${owner}/${repo}/commits?branch=${branch}`,
          {
             headers: {
      Authorization: `token ${token}`,
    },
          }
        );
        setCommits(response.data);
      } catch (err) {
        console.error(err);
        message.error("Lỗi khi lấy danh sách commit");
      } finally {
        setLoading(false);
      }
    };

    setLoading(true);
    fetchCommits();
  }, [owner, repo, branch]);

  if (loading) return <Spin tip="Đang tải commit..." />;

  return (
    <div>
      <Title level={4}>📝 Commit - Branch: {branch}</Title>
      <List
        itemLayout="horizontal"
        dataSource={commits}
        renderItem={(item) => (
          <List.Item>
            <List.Item.Meta
              avatar={<Avatar src={item.author?.avatar_url} />}
              title={item.commit.message}
              description={`Tác giả: ${item.commit.author.name} - ${item.commit.author.date}`}
            />
          </List.Item>
        )}
      />
    </div>
  );
};

export default CommitList;
