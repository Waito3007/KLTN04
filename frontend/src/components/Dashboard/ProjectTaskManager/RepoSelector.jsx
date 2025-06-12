import React from 'react';
import { Select, Avatar, Space, Tag } from 'antd';

const { Option } = Select;

const RepoSelector = ({ repositories, selectedRepo, loading, handleRepoChange }) => (
  <Select
    style={{ width: '100%' }}
    placeholder="Chọn repository để quản lý tasks"
    loading={loading}
    value={selectedRepo?.id}
    onChange={handleRepoChange}
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
);

export default RepoSelector;
