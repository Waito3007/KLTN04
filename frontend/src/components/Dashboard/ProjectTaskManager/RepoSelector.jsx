import React, { useState } from 'react';
import { Select, Avatar, Space, Tag, Button, Card } from 'antd';
import { BranchesOutlined, TeamOutlined, SyncOutlined } from '@ant-design/icons';

const { Option } = Select;

const RepoSelector = ({ 
  repositories, 
  selectedRepo, 
  loading, 
  handleRepoChange,
  branches = [],
  collaborators = [],
  branchesLoading = false,
  onSyncCollaborators = null     
}) => {
  const [isSyncing, setIsSyncing] = useState(false);

  const handleRepoSelect = async (repoId) => {
    await handleRepoChange(repoId);
  };

  const handleSyncClick = async () => {
    if (!onSyncCollaborators || !selectedRepo || isSyncing) return;
    
    setIsSyncing(true);
    try {
      await onSyncCollaborators();
    } catch (error) {
      console.error('Sync failed:', error);
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div>
      <Select
        style={{ width: '100%' }}
        placeholder="Chọn repository để quản lý tasks"
        loading={loading}
        value={selectedRepo?.id}
        onChange={handleRepoSelect}
        showSearch
        optionFilterProp="children"
      >
        {repositories.map(repo => (
          <Option key={repo.id} value={repo.id}>
            <Space>
              <Avatar src={repo.owner.avatar_url} size="small" />
              <span style={{ fontWeight: 500 }}>
                {repo.owner.login}/{repo.name}
              </span>
            </Space>
          </Option>
        ))}
      </Select>

      {selectedRepo && (
        <Card 
          size="small" 
          style={{ 
            marginTop: 16, 
            borderRadius: 8,
            background: 'linear-gradient(135deg, #f6f8fa 0%, #e1e7ed 100%)'
          }}
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            <Space>
              <Avatar src={selectedRepo.owner.avatar_url} size="small" />
              <span style={{ fontWeight: 600, color: '#1890ff' }}>
                {selectedRepo.owner.login}/{selectedRepo.name}
              </span>
              {selectedRepo.private && (
                <Tag color="orange" size="small">Private</Tag>
              )}
            </Space>

            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Space>
                <BranchesOutlined style={{ color: '#52c41a', fontSize: '14px' }} />
                <span style={{ fontSize: '13px', color: '#666', fontWeight: '500' }}>
                  {branchesLoading ? 'Loading...' : `${branches.length} branches`}
                </span>
              </Space>

              <Space>
                <TeamOutlined style={{ color: '#1890ff', fontSize: '14px' }} />
                <span style={{ fontSize: '13px', color: '#666', fontWeight: '500' }}>
                  {collaborators.length} collaborators
                </span>
              </Space>
            </Space>

            {/* Only GitHub sync button - data loads automatically from DB */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'center',
              marginTop: '8px'
            }}>
              <Button
                size="small"
                icon={<SyncOutlined spin={isSyncing} />}
                onClick={handleSyncClick}
                disabled={!selectedRepo || isSyncing}
                type="primary"
                style={{ 
                  background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)',
                  border: 'none'
                }}
              >
                {isSyncing ? 'Đang sync GitHub...' : 'Sync collaborators từ GitHub'}
              </Button>
            </div>

            <div style={{ 
              fontSize: '11px', 
              color: '#999', 
              textAlign: 'center',
              marginTop: '4px'
            }}>
              💡 Dữ liệu tự động tải từ database, chỉ sync GitHub khi cần
            </div>
          </Space>
        </Card>
      )}
    </div>
  );
};

export default RepoSelector;
