import React, { useState } from 'react';
import { List, Card, Button, Alert, Space, Divider, Spin, Input, message } from 'antd';
import { CodeOutlined, SearchOutlined, LinkOutlined } from '@ant-design/icons';

const TaskDetailCommits = ({
  relatedCommits,
  loadingCommits,
  fetchRelatedCommits,
  linkCommitsToTask,
  formatDate
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      message.warning('Vui lòng nhập từ khóa để tìm kiếm commit.');
      return;
    }
    await fetchRelatedCommits(searchTerm);
  };

  return (
    <div style={{ marginTop: 16 }}>
      <Space style={{ marginBottom: 16, width: '100%', justifyContent: 'space-between' }}>
        <Space>
          <CodeOutlined />
          <span>Commits liên quan ({relatedCommits.length})</span>
        </Space>
        <Space>
          <Input
            placeholder="Tìm kiếm commit"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: 200 }}
          />
          <Button
            icon={<SearchOutlined />}
            onClick={handleSearch}
            loading={loadingCommits}
          >
            Tìm kiếm
          </Button>
          <Button
            icon={<LinkOutlined />}
            onClick={linkCommitsToTask}
            loading={loadingCommits}
            type="primary"
          >
            Liên kết Commit
          </Button>
        </Space>
      </Space>

      {loadingCommits ? (
        <Spin />
      ) : relatedCommits.length > 0 ? (
        <List
          dataSource={relatedCommits}
          renderItem={(commit, index) => (
            <List.Item key={commit.sha || index}>
              <Card
                size="small"
                style={{ width: '100%' }}
                title={
                  <Space>
                    <CodeOutlined />
                    <span>{commit.sha?.substring(0, 8)}</span>
                  </Space>
                }
                extra={
                  <Button
                    type="link"
                    size="small"
                    href={commit.url}
                    target="_blank"
                  >
                    Xem chi tiết
                  </Button>
                }
              >
                <div>
                  <strong>{commit.message}</strong>
                </div>
                <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                  <Space split={<Divider type="vertical" />}>
                    <span>Tác giả: {commit.author_name}</span>
                    <span>Ngày: {formatDate(commit.committed_date)}</span>
                    <span style={{ color: '#52c41a' }}>+{commit.insertions}</span>
                    <span style={{ color: '#ff4d4f' }}>-{commit.deletions}</span>
                    <span>{commit.files_changed} files</span>
                  </Space>
                </div>
              </Card>
            </List.Item>
          )}
        />
      ) : (
        <Alert
          message="Không tìm thấy commits liên quan"
          description="Hãy thử sử dụng nút 'Tìm kiếm' để tìm commit liên quan đến task này."
          type="info"
          showIcon
        />
      )}
    </div>
  );
};

export default TaskDetailCommits;
