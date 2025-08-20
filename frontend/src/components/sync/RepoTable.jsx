import React from 'react';
import { Table, Space, Avatar, Tag, Tooltip, Progress, Typography, Button } from 'antd';
import { GithubOutlined, StarFilled, ForkOutlined, SyncOutlined, SettingOutlined, HistoryOutlined } from '@ant-design/icons';

const { Text } = Typography;

const RepoTable = ({ syncStatus, getSyncStatusTag, repoProgresses, debouncedHandleSyncRepository, openSyncModal, showRepoEvents }) => {
  const getColumns = () => [
    {
      title: 'Repository',
      key: 'repo',
      render: (_, repo) => (
        <Space>
          <Avatar 
            size="small" 
            src={`https://github.com/${repo.owner}.png`}
            icon={<GithubOutlined />}
          />
          <div>
            <Text strong>{repo.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {repo.owner}
            </Text>
          </div>
        </Space>
      ),
      width: 200,
    },
    {
      title: 'Mô tả',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (text) => text || <Text type="secondary">Không có mô tả</Text>
    },
    {
      title: 'Ngôn ngữ',
      dataIndex: 'language',
      key: 'language',
      render: (language) => language ? <Tag color="blue">{language}</Tag> : '-',
      width: 100,
    },
    {
      title: 'Thống kê',
      key: 'stats',
      render: (_, repo) => (
        <Space>
          <Tooltip title="Stars">
            <Space size={4}>
              <StarFilled style={{ color: '#faad14' }} />
              <Text>{repo.stars || 0}</Text>
            </Space>
          </Tooltip>
          <Tooltip title="Forks">
            <Space size={4}>
              <ForkOutlined style={{ color: '#1890ff' }} />
              <Text>{repo.forks || 0}</Text>
            </Space>
          </Tooltip>
        </Space>
      ),
      width: 120,
    },
    {
      title: 'Trạng thái',
      key: 'status',
      render: (_, repo) => {
        const repoKey = `${repo.owner}/${repo.name}`;
        const repoProgress = repoProgresses[repoKey];
        
        return (
          <Space direction="vertical" size={4}>
            {getSyncStatusTag(repo)}
            {repo.sync_priority && (
              <Tag 
                color={repo.sync_priority === 'highest' ? '#ff4d4f' : repo.sync_priority === 'high' ? '#faad14' : '#52c41a'}
              >
                {repo.sync_priority.toUpperCase()}
              </Tag>
            )}
            {(repo.sync_status === 'syncing' || repoProgress?.status === 'syncing') && (
              <div style={{ width: '120px' }}>
                <Progress 
                  percent={repoProgress?.progress || repo.sync_progress || 0} 
                  size="small" 
                  status="active"
                  format={(percent) => `${percent}%`}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
                {(repoProgress?.stage || repo.sync_stage) && (
                  <Text style={{ fontSize: '10px', color: '#666', display: 'block', marginTop: 2 }}>
                    {repoProgress?.stage || repo.sync_stage}
                  </Text>
                )}
              </div>
            )}
          </Space>
        );
      },
      width: 180,
    },
    {
      title: 'Thao tác',
      key: 'actions',
      render: (_, repo) => {
        const repoKey = `${repo.owner}/${repo.name}`;
        return (
          <Space>
            <Button
              type="primary"
              size="small"
              icon={<SyncOutlined />}
              loading={repo.syncing}
              onClick={() => debouncedHandleSyncRepository(repo)}
            >
              Đồng bộ
            </Button>
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => openSyncModal(repo)}
            >
              Tùy chọn
            </Button>
            <Button
              size="small"
              icon={<HistoryOutlined />}
              onClick={() => showRepoEvents(repo)}
            >
              Sự kiện
            </Button>
          </Space>
        );
      },
      width: 200,
    },
  ];

  const columns = getColumns();
  if (!Array.isArray(columns)) {
    console.error('Columns is not an array:', columns);
    return null;
  }

  return (
    <Table
      columns={columns}
      dataSource={syncStatus?.repositories?.unsynced || []}
      rowKey={(record) => `${record.owner}-${record.name}`}
      pagination={{ pageSize: 10 }}
      scroll={{ x: 1000 }}
    />
  );
};

export default RepoTable;
