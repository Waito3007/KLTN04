import { useState } from "react";
import { Avatar, Typography, Tag, Pagination } from "antd";
import { useNavigate } from "react-router-dom";
import { GithubOutlined, StarFilled, EyeFilled, ForkOutlined, CalendarOutlined } from "@ant-design/icons";
import styled from "styled-components";
import { Loading, Toast, Card } from "@components/common";

const { Title, Text } = Typography;

const RepoContainer = styled.div`
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
`;

const RepoHeader = styled.div`
  display: flex;
  align-items: flex-start;
  margin-bottom: 12px;
`;

const RepoTitle = styled.div`
  flex: 1;
  min-width: 0;
`;

const RepoName = styled(Text)`
  display: block;
  font-size: 18px;
  font-weight: 600;
  color: #24292e;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const RepoDescription = styled(Text)`
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  color: #586069;
  margin: 8px 0;
`;

const RepoMeta = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 16px;
  align-items: center;
`;

const MetaItem = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  color: #586069;
`;

const StyledPagination = styled(Pagination)`
  margin-top: 32px;
  text-align: center;
  
  .ant-pagination-item-active {
    border-color: #1890ff;
    background: #1890ff;
    
    a {
      color: white;
    }
  }
`;

const HighlightTag = styled(Tag)`
  font-weight: 500;
  border-radius: 12px;
  padding: 0 10px;
`;

const RepoList = ({ repositories = [], onRepoClick }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const navigate = useNavigate();
  const pageSize = 8;

  // Use provided repositories instead of fetching
  const repos = repositories;
  const totalRepos = repositories.length;
  const loading = false;

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('vi-VN', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric'
    });
  };

  const paginatedRepos = repos.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );
  if (loading) {
    return (
      <Loading 
        variant="modern"
        text="Đang tải dữ liệu repository..."
        size="large"
      />
    );
  }

  return (
    <RepoContainer>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#24292e' }}>
          <GithubOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          GitHub Repositories
        </Title>
        <Text strong style={{ fontSize: '16px' }}>
          Tổng cộng: {totalRepos} repositories
        </Text>
      </div>

      {paginatedRepos.map((repo) => (
        <Card 
          key={repo.id} 
          variant="modern"
          hoverable={true}
          style={{ 
            marginBottom: '20px',
            cursor: 'pointer'
          }}
          onClick={() => {
            if (onRepoClick) {
              onRepoClick(repo);
            } else {
              navigate(`/repo/${repo.owner.login}/${repo.name}`);
            }
          }}
        >
          <RepoHeader>
            <Avatar 
              src={repo.owner.avatar_url} 
              size={48}
              style={{ marginRight: '16px', flexShrink: 0 }}
            />
            <RepoTitle>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <RepoName>{repo.name}</RepoName>
                {repo.private ? (
                  <HighlightTag color="error" style={{ marginLeft: '12px' }}>
                    Private
                  </HighlightTag>
                ) : (
                  <HighlightTag color="success" style={{ marginLeft: '12px' }}>
                    Public
                  </HighlightTag>
                )}
              </div>
              
              <RepoDescription type="secondary">
                {repo.description || "Không có mô tả"}
              </RepoDescription>
            </RepoTitle>
          </RepoHeader>

          <RepoMeta>
            <MetaItem>
              <StarFilled style={{ color: '#ffc53d' }} />
              <Text strong>{repo.stargazers_count}</Text>
              <Text>stars</Text>
            </MetaItem>
            
            <MetaItem>
              <EyeFilled style={{ color: '#1890ff' }} />
              <Text strong>{repo.watchers_count}</Text>
              <Text>watchers</Text>
            </MetaItem>
            
            <MetaItem>
              <ForkOutlined style={{ color: '#73d13d' }} />
              <Text strong>{repo.forks_count}</Text>
              <Text>forks</Text>
            </MetaItem>
            
            {repo.language && (
              <MetaItem>
                <div style={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: '#1890ff',
                  marginRight: 6
                }} />
                <Text>{repo.language}</Text>
              </MetaItem>
            )}
            
            <MetaItem style={{ marginLeft: 'auto' }}>
              <CalendarOutlined />
              <Text>Cập nhật: {formatDate(repo.updated_at)}</Text>
            </MetaItem>
          </RepoMeta>
        </Card>
      ))}

      <StyledPagination
        current={currentPage}
        pageSize={pageSize}
        total={totalRepos}
        onChange={(page) => setCurrentPage(page)}
        showSizeChanger={false}
        showQuickJumper
      />
    </RepoContainer>
  );
};

export default RepoList;