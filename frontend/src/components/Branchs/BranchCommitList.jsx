import React, { useState, useEffect } from 'react';
import { Card, List, Tag, Typography, Button, Space, message, Spin, Empty, Modal } from 'antd';
import { 
  GitlabOutlined, 
  UserOutlined, 
  CalendarOutlined, 
  FileTextOutlined,
  EyeOutlined,
  BranchesOutlined,
  PlusOutlined,
  MinusOutlined,
  FileOutlined
} from '@ant-design/icons';
import axios from 'axios';
import styled from 'styled-components';

const { Text, Paragraph } = Typography;

const CommitCard = styled(Card)`
  margin-bottom: 12px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;

  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
  }

  .ant-card-body {
    padding: 16px;
  }
`;

const CommitHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
`;

const CommitMessage = styled(Paragraph)`
  font-weight: 500;
  color: #262626;
  margin-bottom: 8px;
  
  &.ant-typography {
    margin-bottom: 8px;
  }
`;

const CommitMeta = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  color: #8c8c8c;
  font-size: 12px;
  flex-wrap: wrap;
`;

const StatsContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
`;

const StatTag = styled(Tag)`
  margin: 0;
  display: flex;
  align-items: center;
  gap: 4px;
`;

const BranchCommitList = ({ owner, repo, selectedBranch }) => {
  const [commits, setCommits] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState({ current: 1, pageSize: 10, total: 0 });
  const [selectedCommit, setSelectedCommit] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  useEffect(() => {
    if (selectedBranch) {
      fetchCommits();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBranch, pagination.current]);

  const fetchCommits = async () => {
    if (!selectedBranch) return;

    const token = localStorage.getItem("access_token");
    if (!token) {
      message.error("Vui lòng đăng nhập lại!");
      return;
    }

    setLoading(true);
    try {
      const offset = (pagination.current - 1) * pagination.pageSize;
      const response = await axios.get(
        `http://localhost:8000/api/commits/${owner}/${repo}/branches/${selectedBranch}/commits?limit=${pagination.pageSize}&offset=${offset}`,
        {
          headers: {
            Authorization: `token ${token}`,
          },
        }
      );

      setCommits(response.data.commits || []);
      setPagination(prev => ({
        ...prev,
        total: response.data.total_found || response.data.count || 0
      }));

    } catch (error) {
      console.error("Error fetching commits:", error);
      if (error.response?.status === 404) {
        message.warning(`Chưa có commits nào trong database cho branch "${selectedBranch}". Hãy đồng bộ dữ liệu trước!`);
        setCommits([]);
      } else {
        message.error("Không thể lấy danh sách commits!");
      }
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString('vi-VN');
    } catch {
      return dateString;
    }
  };

  const getCommitType = (message) => {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.startsWith('feat:') || lowerMessage.includes('feature')) {
      return { type: 'feat', color: 'blue' };
    } else if (lowerMessage.startsWith('fix:') || lowerMessage.includes('bug')) {
      return { type: 'fix', color: 'red' };
    } else if (lowerMessage.startsWith('docs:')) {
      return { type: 'docs', color: 'green' };
    } else if (lowerMessage.startsWith('style:')) {
      return { type: 'style', color: 'purple' };
    } else if (lowerMessage.startsWith('refactor:')) {
      return { type: 'refactor', color: 'orange' };
    } else if (lowerMessage.startsWith('test:')) {
      return { type: 'test', color: 'cyan' };
    } else {
      return { type: 'other', color: 'default' };
    }
  };

  const showCommitDetails = (commit) => {
    setSelectedCommit(commit);
    setModalVisible(true);
  };

  const handlePageChange = (page) => {
    setPagination(prev => ({ ...prev, current: page }));
  };

  if (!selectedBranch) {
    return (
      <Card>
        <Empty 
          description="Vui lòng chọn branch để xem commits"
          image={<BranchesOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />}
        />
      </Card>
    );
  }

  return (
    <>
      <Card 
        title={
          <Space>
            <GitlabOutlined style={{ color: '#1890ff' }} />
            <span>Commits - Branch: {selectedBranch}</span>
            <Tag color="blue">{commits.length} commits</Tag>
          </Space>
        }
        extra={
          <Button 
            type="primary" 
            size="small" 
            onClick={fetchCommits}
            loading={loading}
          >
            Làm mới
          </Button>
        }
      >
        <Spin spinning={loading}>
          {commits.length === 0 && !loading ? (
            <Empty 
              description={`Chưa có commits nào cho branch "${selectedBranch}"`}
              image={<GitlabOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />}
            />
          ) : (
            <List
              itemLayout="vertical"
              dataSource={commits}
              pagination={{
                current: pagination.current,
                pageSize: pagination.pageSize,
                total: pagination.total,
                onChange: handlePageChange,
                showSizeChanger: false,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `${range[0]}-${range[1]} của ${total} commits`,
              }}
              renderItem={(commit) => {
                const commitType = getCommitType(commit.message);
                
                return (
                  <List.Item key={commit.sha}>
                    <CommitCard size="small">
                      <CommitHeader>
                        <div style={{ flex: 1 }}>
                          <Space size="small">
                            <Tag color={commitType.color}>{commitType.type}</Tag>
                            <Text code style={{ fontSize: '11px' }}>
                              {commit.sha?.substring(0, 7)}
                            </Text>
                          </Space>
                        </div>
                        <Button
                          type="text"
                          size="small"
                          icon={<EyeOutlined />}
                          onClick={() => showCommitDetails(commit)}
                        >
                          Chi tiết
                        </Button>
                      </CommitHeader>

                      <CommitMessage ellipsis={{ rows: 2, expandable: true }}>
                        {commit.message}
                      </CommitMessage>

                      <CommitMeta>
                        <Space size={4}>
                          <UserOutlined />
                          <Text>{commit.author_name}</Text>
                        </Space>
                        <Space size={4}>
                          <CalendarOutlined />
                          <Text>{formatDate(commit.date)}</Text>
                        </Space>
                        {commit.branch_name && (
                          <Space size={4}>
                            <BranchesOutlined />
                            <Text>{commit.branch_name}</Text>
                          </Space>
                        )}
                      </CommitMeta>

                      {(commit.insertions || commit.deletions || commit.files_changed) && (
                        <StatsContainer>
                          {commit.insertions && (
                            <StatTag color="green">
                              <PlusOutlined />
                              {commit.insertions}
                            </StatTag>
                          )}
                          {commit.deletions && (
                            <StatTag color="red">
                              <MinusOutlined />
                              {commit.deletions}
                            </StatTag>
                          )}
                          {commit.files_changed && (
                            <StatTag color="blue">
                              <FileOutlined />
                              {commit.files_changed} files
                            </StatTag>
                          )}
                        </StatsContainer>
                      )}
                    </CommitCard>
                  </List.Item>
                );
              }}
            />
          )}
        </Spin>
      </Card>

      <Modal
        title={
          <Space>
            <GitlabOutlined />
            <span>Chi tiết Commit</span>
            {selectedCommit && (
              <Text code>{selectedCommit.sha?.substring(0, 7)}</Text>
            )}
          </Space>
        }
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={600}
      >
        {selectedCommit && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>SHA:</Text>
              <br />
              <Text code>{selectedCommit.sha}</Text>
            </div>

            <div style={{ marginBottom: 16 }}>
              <Text strong>Message:</Text>
              <br />
              <Paragraph>{selectedCommit.message}</Paragraph>
            </div>

            <div style={{ marginBottom: 16 }}>
              <Text strong>Author:</Text>
              <br />
              <Text>{selectedCommit.author_name} ({selectedCommit.author_email})</Text>
            </div>

            <div style={{ marginBottom: 16 }}>
              <Text strong>Date:</Text>
              <br />
              <Text>{formatDate(selectedCommit.date)}</Text>
            </div>

            {selectedCommit.committer_name && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>Committer:</Text>
                <br />
                <Text>{selectedCommit.committer_name} ({selectedCommit.committer_email})</Text>
              </div>
            )}

            {(selectedCommit.insertions || selectedCommit.deletions || selectedCommit.files_changed) && (
              <div style={{ marginBottom: 16 }}>
                <Text strong>Statistics:</Text>
                <br />
                <Space>
                  {selectedCommit.insertions && (
                    <Tag color="green">+{selectedCommit.insertions} insertions</Tag>
                  )}
                  {selectedCommit.deletions && (
                    <Tag color="red">-{selectedCommit.deletions} deletions</Tag>
                  )}
                  {selectedCommit.files_changed && (
                    <Tag color="blue">{selectedCommit.files_changed} files changed</Tag>
                  )}
                </Space>
              </div>
            )}

            {selectedCommit.is_merge && (
              <div style={{ marginBottom: 16 }}>
                <Tag color="purple">Merge Commit</Tag>
                {selectedCommit.merge_from_branch && (
                  <Text> từ branch: {selectedCommit.merge_from_branch}</Text>
                )}
              </div>
            )}
          </div>
        )}
      </Modal>
    </>
  );
};

export default BranchCommitList;
