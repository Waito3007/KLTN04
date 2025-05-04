import { useEffect, useState } from "react";
import { List, Avatar, Typography, Spin, message, Tooltip, Card, Tag, Pagination } from "antd";
import { GithubOutlined, BranchesOutlined, ClockCircleOutlined, UserOutlined } from '@ant-design/icons';
import axios from "axios";
import styled from "styled-components";

const { Title, Text } = Typography;

const CommitCard = styled(Card)`
  margin-bottom: 16px;
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
  }
`;

const CommitHeader = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
`;

const CommitMessage = styled.div`
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
  
  &:hover {
    white-space: normal;
    overflow: visible;
  }
`;

const CommitMeta = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 8px;
  color: #666;
  font-size: 13px;
`;

const PaginationContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 20px;
`;

const CommitList = ({ owner, repo, branch }) => {
  const [commits, setCommits] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 5;

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

  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString('vi-VN', options);
  };

  // Tính toán dữ liệu hiển thị theo trang hiện tại
  const paginatedCommits = commits.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  if (loading) return <Spin tip="Đang tải commit..." size="large" />;

  return (
    <div style={{ padding: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
        <Title level={4} style={{ margin: 0 }}>
          <BranchesOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          Commit trên branch: <Tag color="blue">{branch}</Tag>
          <Tag style={{ marginLeft: '8px' }}>{commits.length} commits</Tag>
        </Title>
      </div>
      
      <List
        itemLayout="vertical"
        dataSource={paginatedCommits}
        renderItem={(item) => (
          <List.Item>
            <CommitCard>
              <CommitHeader>
                <Tooltip title={item.sha} placement="topLeft">
                  <Tag icon={<GithubOutlined />} color="default">
                    {item.sha.substring(0, 7)}
                  </Tag>
                </Tooltip>
              </CommitHeader>
              
              <CommitMessage>
                <Tooltip title={item.commit.message} placement="topLeft">
                  {item.commit.message.split('\n')[0]}
                </Tooltip>
              </CommitMessage>
              
              <CommitMeta>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <Avatar 
                    src={item.author?.avatar_url} 
                    size="small" 
                    icon={<UserOutlined />}
                    style={{ marginRight: '8px' }}
                  />
                  <Text>{item.commit.author.name}</Text>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <ClockCircleOutlined style={{ marginRight: '4px' }} />
                  <Text>{formatDate(item.commit.author.date)}</Text>
                </div>
              </CommitMeta>
            </CommitCard>
          </List.Item>
        )}
      />

      <PaginationContainer>
        <Pagination
          current={currentPage}
          pageSize={pageSize}
          total={commits.length}
          onChange={(page) => setCurrentPage(page)}
          showSizeChanger={false}
          showQuickJumper
          style={{ marginTop: '20px' }}
        />
      </PaginationContainer>
    </div>
  );
};

export default CommitList;