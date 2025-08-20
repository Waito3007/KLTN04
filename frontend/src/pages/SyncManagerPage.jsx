import React, { useState } from 'react';
import PageLayout from '@components/common/PageLayout';
import SyncSummary from '@components/sync/SyncSummary';
import SyncTabs from '@components/sync/SyncTabs';
import GlobalSyncProgress from '@components/sync/GlobalSyncProgress';
import SyncModal from '@components/sync/SyncModal';
import SyncEventsDrawer from '@components/sync/SyncEventsDrawer';
import useSyncStatus from '@hooks/useSyncStatus';
import useWebSocket from '@hooks/useWebSocket';
import { syncRepository, getRepoEvents } from '@services/syncService';
import { notification } from 'antd';
import { Avatar, Button, Space, Tag, Tooltip, Progress } from 'antd';
import { GithubOutlined, SyncOutlined, SettingOutlined, StarFilled, ForkOutlined } from '@ant-design/icons';

const SyncManagerPage = () => {
  const { syncStatus, fetchSyncStatus } = useSyncStatus();
  const [activeTab, setActiveTab] = useState('unsynced');
  const [syncModalVisible, setSyncModalVisible] = useState(false);
  const [syncType, setSyncType] = useState('optimized');
  const [eventsDrawerVisible, setEventsDrawerVisible] = useState(false);
  const [selectedRepoEvents, setSelectedRepoEvents] = useState(null);
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [globalSyncProgress, setGlobalSyncProgress] = useState({
    visible: false,
    totalRepos: 0,
    completedRepos: 0,
    currentRepo: '',
    overallProgress: 0,
  });

  useWebSocket('/api/sync-events/ws', (event) => {
    console.log('WebSocket event:', event);
    if (event.type === 'progress_update') {
      setGlobalSyncProgress((prev) => ({
        ...prev,
        [event.repoKey]: event.progress,
      }));
    }
    fetchSyncStatus(); // Cập nhật trạng thái đồng bộ khi nhận sự kiện mới
  });

  const handleSyncRepository = async (repo) => {
    const repoKey = `${repo.owner}/${repo.name}`;
    try {
      await syncRepository(repoKey);
      notification.success({
        message: 'Thành công',
        description: `Đã đồng bộ repository ${repoKey}.`,
      });
      fetchSyncStatus();
    } catch {
      notification.error({
        message: 'Lỗi',
        description: `Không thể đồng bộ repository ${repoKey}.`,
      });
    }
  };

  const showRepoEvents = async (repo) => {
    try {
      const data = await getRepoEvents(repo.owner, repo.name);
      setSelectedRepoEvents(data);
      setEventsDrawerVisible(true);
    } catch {
      notification.error({
        message: 'Lỗi',
        description: 'Không thể tải sự kiện repository.',
      });
    }
  };

  const openSyncModal = (repo) => {
    setSelectedRepo(repo); // Sử dụng setSelectedRepo để tránh lỗi lint
    setSyncModalVisible(true);
  };

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
        const repoProgress = globalSyncProgress[repoKey];
        return (
          <Space direction="vertical" size={4}>
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
      render: (_, repo) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<SyncOutlined />}
            onClick={() => handleSyncRepository(repo)}
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
        </Space>
      ),
      width: 200,
    },
  ];

  return (
    <PageLayout>
      <SyncSummary summary={syncStatus?.summary} />
      <GlobalSyncProgress globalSyncProgress={globalSyncProgress ?? {
        visible: false,
        totalRepos: 0,
        completedRepos: 0,
        currentRepo: '',
        overallProgress: 0,
      }} />
      <SyncTabs
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        syncStatus={syncStatus}
        getColumns={getColumns} // Truyền hàm getColumns
        handleSyncRepository={handleSyncRepository}
        showRepoEvents={showRepoEvents}
      />
      <SyncModal
        syncModalVisible={syncModalVisible}
        setSyncModalVisible={setSyncModalVisible}
        syncType={syncType}
        setSyncType={setSyncType}
        handleModalSync={() => handleSyncRepository(selectedRepo)}
      />
      <SyncEventsDrawer
        eventsDrawerVisible={eventsDrawerVisible}
        setEventsDrawerVisible={setEventsDrawerVisible}
        selectedRepoEvents={selectedRepoEvents}
      />
    </PageLayout>
  );
};

export default SyncManagerPage;
