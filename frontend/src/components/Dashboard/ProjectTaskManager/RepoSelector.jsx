import React from 'react';
import { Select, Avatar, Space, Tag, Spin, Badge } from 'antd';
import { BranchesOutlined, TeamOutlined, SyncOutlined } from '@ant-design/icons';

const { Option } = Select;

const RepoSelector = ({ 
  repositories, 
  selectedRepo, 
  loading, 
  handleRepoChange,
  branches = [],
  collaborators = [],
  branchesLoading = false
}) => {

  const handleRepoSelect = async (repoId) => {
    await handleRepoChange(repoId);
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
              {repo.owner.login}/{repo.name}
              <Tag color={repo.private ? 'red' : 'green'}>
                {repo.private ? 'Private' : 'Public'}
              </Tag>
            </Space>
          </Option>
        ))}
      </Select>

      {/* Repository sync status and stats */}
      {selectedRepo && (
        <div style={{ marginTop: 12, padding: '8px 12px', background: '#f5f5f5', borderRadius: 6 }}>
          <Space>
            {branchesLoading ? (
              <>
                <Spin size="small" />
                <span style={{ color: '#1890ff' }}>
                  <SyncOutlined spin /> Đang đồng bộ...
                </span>
              </>
            ) : (
              <>
                <Badge count={branches.length} showZero color="#52c41a">
                  <BranchesOutlined style={{ color: '#52c41a' }} />
                </Badge>
                <span style={{ fontSize: '12px', color: '#666' }}>
                  {branches.length} branches
                </span>
                
                <Badge count={collaborators.length} showZero color="#1890ff">
                  <TeamOutlined style={{ color: '#1890ff' }} />
                </Badge>
                <span style={{ fontSize: '12px', color: '#666' }}>
                  {collaborators.length} collaborators
                </span>
              </>
            )}
          </Space>
        </div>
      )}
    </div>
  );
};

export default RepoSelector;
