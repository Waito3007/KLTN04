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
  onSyncCollaborators = null,
  onSyncBranches = null     
}) => {
  const [isSyncingCollaborators, setIsSyncingCollaborators] = useState(false);
  const [isSyncingBranches, setIsSyncingBranches] = useState(false);
  const handleRepoSelect = async (repoId) => {
    await handleRepoChange(repoId);
  };

  const handleSyncCollaborators = async () => {
    if (!onSyncCollaborators || !selectedRepo || isSyncingCollaborators) return;
    
    setIsSyncingCollaborators(true);
    try {
      await onSyncCollaborators();
    } catch (error) {
      console.error('Sync collaborators failed:', error);
    } finally {
      setIsSyncingCollaborators(false);
    }
  };

  const handleSyncBranches = async () => {
    if (!onSyncBranches || !selectedRepo || isSyncingBranches) return;
    
    setIsSyncingBranches(true);
    try {
      await onSyncBranches();
    } catch (error) {
      console.error('Sync branches failed:', error);
    } finally {
      setIsSyncingBranches(false);
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
      >        {repositories.map(repo => (
          <Option key={repo.id} value={repo.id}>
            <Space>
              <Avatar 
                src={repo.owner?.avatar_url} 
                size="small"
                style={{ backgroundColor: '#1890ff' }}
              >
                {!repo.owner?.avatar_url && repo.owner?.login?.charAt(0)?.toUpperCase()}
              </Avatar>
              <span style={{ fontWeight: 500 }}>
                {repo.owner?.login}/{repo.name}
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
          <Space direction="vertical" style={{ width: '100%' }}>            <Space>
              <Avatar 
                src={selectedRepo.owner?.avatar_url} 
                size="small"
                style={{ backgroundColor: '#1890ff' }}
              >
                {!selectedRepo.owner?.avatar_url && selectedRepo.owner?.login?.charAt(0)?.toUpperCase()}
              </Avatar>
              <span style={{ fontWeight: 600, color: '#1890ff' }}>
                {selectedRepo.owner?.login}/{selectedRepo.name}
              </span>
              {selectedRepo.private && (
                <Tag color="orange" size="small">Private</Tag>
              )}
            </Space>            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Space>
                <BranchesOutlined style={{ color: '#52c41a', fontSize: '14px' }} />
                <span style={{ fontSize: '13px', color: '#666', fontWeight: '500' }}>
                  {branchesLoading ? 'Loading...' : `${branches.length} branches`}
                </span>
                {onSyncBranches && (
                  <Button
                    size="small"
                    type="text"
                    icon={<SyncOutlined spin={isSyncingBranches} />}
                    onClick={handleSyncBranches}
                    disabled={!selectedRepo || isSyncingBranches}
                    style={{ 
                      fontSize: '12px',
                      height: '20px',
                      padding: '0 4px',
                      color: '#52c41a'
                    }}
                    title="Sync branches từ GitHub"
                  />
                )}
              </Space>

              <Space>
                <TeamOutlined style={{ color: '#1890ff', fontSize: '14px' }} />
                <span style={{ fontSize: '13px', color: '#666', fontWeight: '500' }}>
                  {collaborators.length} collaborators
                </span>
                {onSyncCollaborators && (
                  <Button
                    size="small"
                    type="text"
                    icon={<SyncOutlined spin={isSyncingCollaborators} />}
                    onClick={handleSyncCollaborators}
                    disabled={!selectedRepo || isSyncingCollaborators}
                    style={{ 
                      fontSize: '12px',
                      height: '20px',
                      padding: '0 4px',
                      color: '#1890ff'
                    }}
                    title="Sync collaborators từ GitHub"
                  />
                )}
              </Space>
            </Space>            <div style={{ 
              fontSize: '11px', 
              color: '#999', 
              textAlign: 'center',
              marginTop: '8px'
            }}>
              
            </div>
          </Space>
        </Card>
      )}
    </div>
  );
};

export default RepoSelector;
