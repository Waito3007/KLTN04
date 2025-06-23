import { useEffect, useState, useCallback } from "react";
import { List, Avatar, Typography, Spin, message, Tooltip, Card, Tag, Pagination, Select } from "antd";
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
  justify-content: space-between;
  align-items: center;
  margin-top: 20px;
  padding: 0 16px;
`;

const PageSizeSelector = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const CommitList = ({ owner, repo, branch }) => {
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(30);
  const [totalCommits, setTotalCommits] = useState(0);
  const [allCommits, setAllCommits] = useState([]); // Lưu tất cả commits để phân trang client-side// Function để fetch commits từ API - lấy nhiều commits để phân trang client-side
  const fetchCommits = useCallback(async (requestedPageSize = pageSize) => {
    const token = localStorage.getItem("access_token");
    if (!token) return;

    try {
      setLoading(true);
      // Lấy số lượng commits lớn hơn để có đủ dữ liệu phân trang
      const fetchSize = Math.max(requestedPageSize, 100); // Ít nhất 100 commits
      
      const response = await axios.get(
        `http://localhost:8000/api/github/${owner}/${repo}/branches/${encodeURIComponent(branch)}/commits`,
        {
          headers: {
            Authorization: `token ${token}`,
          },
          params: {
            per_page: fetchSize,
            page: 1 // Luôn lấy từ trang 1
          }
        }
      );
      
      const commitsData = response.data.commits || response.data;
      const newCommits = Array.isArray(commitsData) ? commitsData : [];
      
      setAllCommits(newCommits);
      setTotalCommits(newCommits.length);
        console.log(`Fetched ${newCommits.length} commits from GitHub API`);
      
    } catch (err) {
      console.error(err);
      message.error("Lỗi khi lấy danh sách commit");
      setAllCommits([]);
      setTotalCommits(0);
    } finally {
      setLoading(false);
    }
  }, [owner, repo, branch, pageSize]);useEffect(() => {
    if (!branch) return;
    
    setCurrentPage(1);
    fetchCommits(pageSize);
  }, [owner, repo, branch, pageSize, fetchCommits]);const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString('vi-VN', options);
  };
  // Xử lý thay đổi trang - chỉ thay đổi currentPage (client-side pagination)
  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  // Xử lý thay đổi số lượng commits trên trang
  const handlePageSizeChange = (newPageSize) => {
    setPageSize(newPageSize);
    setCurrentPage(1);
    // Nếu cần thêm dữ liệu, fetch lại
    if (newPageSize > allCommits.length) {
      fetchCommits(newPageSize);
    }
  };

  // Tính toán dữ liệu hiển thị theo trang hiện tại (client-side pagination)
  const paginatedCommits = Array.isArray(allCommits) ? allCommits.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  ) : [];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>Đang tải commit...</Text>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '16px' }}>      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>        <Title level={4} style={{ margin: 0, flex: 1 }}>
          <BranchesOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          Commit trên branch: <Tag color="blue">{branch}</Tag>
          <Tag style={{ marginLeft: '8px' }}>{totalCommits} commits</Tag>
        </Title>
        
        <PageSizeSelector>
          <Text>Hiển thị:</Text>
          <Select
            value={pageSize}
            onChange={handlePageSizeChange}
            style={{ width: 80 }}
            options={[
              { value: 10, label: '10' },
              { value: 20, label: '20' },
              { value: 30, label: '30' },
              { value: 50, label: '50' },
              { value: 100, label: '100' }
            ]}
          />
          <Text>commits</Text>
        </PageSizeSelector>
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
                <Tooltip title={item.message || item.commit?.message} placement="topLeft">
                  {(item.message || item.commit?.message || '').split('\n')[0]}
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
                  <Text>{item.author_name || item.commit?.author?.name || 'Unknown'}</Text>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <ClockCircleOutlined style={{ marginRight: '4px' }} />
                  <Text>{formatDate(item.author_date || item.commit?.author?.date)}</Text>
                </div>
              </CommitMeta>
            </CommitCard>
          </List.Item>
        )}
      />      <PaginationContainer>
        <div>
          <Text type="secondary">
            Hiển thị {paginatedCommits.length} trong tổng số {totalCommits} commits
            (Trang {currentPage} / {Math.ceil(totalCommits / pageSize)})
          </Text>
        </div>
        
        <Pagination
          current={currentPage}
          pageSize={pageSize}
          total={totalCommits}
          onChange={handlePageChange}
          showSizeChanger={false}
          showQuickJumper
          showTotal={(total, range) => 
            `${range[0]}-${range[1]} của ${total} commits`
          }
        />
      </PaginationContainer>
    </div>
  );
};

export default CommitList;