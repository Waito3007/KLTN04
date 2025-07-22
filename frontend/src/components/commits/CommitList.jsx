import { useEffect, useState, useCallback } from "react";
import { List, Avatar, Typography, Spin, message, Tooltip, Card, Tag, Pagination, Select, Empty } from "antd";
import { GithubOutlined, BranchesOutlined, ClockCircleOutlined, UserOutlined } from '@ant-design/icons';
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

const CommitList = ({
  memberCommits,
  selectedMember,
  selectedBranch,
  commitTypeFilter,
  setCommitTypeFilter,
  techAreaFilter,
  setTechAreaFilter,
  currentPage,
  setCurrentPage,
  pageSize,
  allRepoCommits // NEW PROP
}) => {
  const [loading, setLoading] = useState(false); // Loading state is now managed by parent
  const [paginatedCommits, setPaginatedCommits] = useState([]);
  const [totalCommits, setTotalCommits] = useState(0);

  useEffect(() => {
    let commitsToFilter = [];

    if (selectedMember) {
      // Member-specific commits
      if (memberCommits && memberCommits.commits) {
        commitsToFilter = memberCommits.commits;
      }
    } else if (allRepoCommits) {
      // All repo commits
      commitsToFilter = allRepoCommits;
    }

    const filteredCommits = commitsToFilter.filter(commit => {
      const matchesType = commitTypeFilter === 'all' || 
                            (commit.analysis && commit.analysis.type === commitTypeFilter);
      const matchesTechArea = techAreaFilter === 'all' || 
                              (commit.analysis && commit.analysis.tech_area === techAreaFilter);
      return matchesType && matchesTechArea;
    });

    setTotalCommits(filteredCommits.length);
    setPaginatedCommits(filteredCommits.slice(
      (currentPage - 1) * pageSize,
      currentPage * pageSize
    ));
  }, [memberCommits, currentPage, pageSize, commitTypeFilter, techAreaFilter, selectedMember, allRepoCommits]);

  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString('vi-VN', options);
  };
  // Xử lý thay đổi trang - chỉ thay đổi currentPage (client-side pagination)
  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  // Xử lý thay đổi số lượng commits trên trang
  const handlePageSizeChange = (newPageSize) => {
    // pageSize is now a prop, so we need to call setPageSize from parent
    // This function might not be needed here if pageSize is controlled by parent
    // For now, we'll just update currentPage
    setCurrentPage(1);
  };

  if (!selectedMember && (!allRepoCommits || allRepoCommits.length === 0)) {
    return (
      <Card>
        <Empty description="Không có commit nào được tìm thấy cho repository này." />
      </Card>
    );
  }

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

  const commitsToDisplay = selectedMember ? (memberCommits?.commits || []) : (allRepoCommits || []);

  if (commitsToDisplay.length === 0) {
    return (
      <Card>
        <Empty description="Không có commit nào được tìm thấy." />
      </Card>
    );
  }

  const displayTitle = selectedMember 
    ? `Commit của ${selectedMember.login} trên nhánh: ${selectedBranch || "Tất cả"}`
    : `Tất cả Commit trên nhánh: ${selectedBranch || "Tất cả"}`;

  return (
    <div style={{ padding: '16px' }}>      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>        <Title level={4} style={{ margin: 0, flex: 1 }}>
          <BranchesOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          {displayTitle}
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
                <Tooltip title={item.message} placement="topLeft">
                  {(item.message || '').split('\n')[0]}
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
                  <Text>{item.author_name || 'Unknown'}</Text>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <ClockCircleOutlined style={{ marginRight: '4px' }} />
                  <Text>{formatDate(item.date)}</Text>
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