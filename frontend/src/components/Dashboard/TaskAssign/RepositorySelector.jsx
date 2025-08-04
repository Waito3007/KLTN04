/**
 * RepositorySelector - Component chọn repository
 * Tuân thủ quy tắc KLTN04: Defensive programming, immutability
 */

import React, { useMemo } from 'react';
import { Select, Space, Typography, Avatar } from 'antd';
import { GithubOutlined, FolderOutlined } from '@ant-design/icons';
import PropTypes from 'prop-types';

const { Option } = Select;
const { Text } = Typography;

const RepositorySelector = ({ 
  repositories = [], 
  loading = false, 
  selectedRepo = null, 
  onRepositoryChange, 
  placeholder = "Chọn repository" 
}) => {
  console.log('🔍 RepositorySelector render:', { 
    selectedRepo, 
    repositoriesCount: repositories.length,
    selectedRepoId: selectedRepo?.id 
  });
  // Defensive programming: Validate và format dữ liệu repositories
  const validRepositories = useMemo(() => {
    if (!Array.isArray(repositories)) return [];
    
    return repositories
      .filter(repo => repo && repo.owner && repo.name)
      .map(repo => {
        // Handle owner có thể là string hoặc object
        const ownerName = typeof repo.owner === 'string' 
          ? repo.owner 
          : repo.owner?.login || repo.owner?.name || 'unknown';
        
        return {
          ...repo, // GIỮ NGUYÊN cấu trúc repo gốc
          key: `${ownerName}/${repo.name}`,
          displayName: `${ownerName}/${repo.name}`,
          description: repo.description || 'Không có mô tả',
          ownerName // Thêm ownerName để display, nhưng giữ nguyên repo.owner
        };
      });
  }, [repositories]);

  // Handler với error handling
  const handleChange = (value) => {
    try {
      if (!value) {
        onRepositoryChange?.(null);
        return;
      }

      // Tìm repository bằng ID thay vì key
      const selectedRepository = validRepositories.find(repo => repo.id === value);
      console.log('🔄 RepositorySelector: Selected repo by ID:', value, selectedRepository);
      if (selectedRepository) {
        onRepositoryChange?.(selectedRepository);
      }
    } catch (error) {
      console.error('Lỗi khi chọn repository:', error);
    }
  };

  // Custom option renderer
  const renderOption = (repo) => (
    <Option key={repo.id} value={repo.id}>
      <div style={{ display: 'flex', alignItems: 'center', padding: '4px 0' }}>
        <Avatar
          size="small"
          icon={<GithubOutlined />}
          style={{ 
            backgroundColor: '#1890ff', 
            marginRight: 8,
            flexShrink: 0
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 500, color: '#262626' }}>
            {repo.displayName}
          </div>
          {repo.description && (
            <div 
              style={{ 
                fontSize: '12px', 
                color: '#8c8c8c',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
            >
              {repo.description}
            </div>
          )}
        </div>
        {repo.private && (
          <Text 
            style={{ 
              fontSize: '11px', 
              color: '#fa8c16',
              marginLeft: 8,
              flexShrink: 0
            }}
          >
            Private
          </Text>
        )}
      </div>
    </Option>
  );

  return (
    <Select
      style={{ width: '100%' }}
      placeholder={
        <Space>
          <FolderOutlined />
          {placeholder}
        </Space>
      }
      loading={loading}
      value={selectedRepo?.id || undefined}
      onChange={handleChange}
      showSearch
      filterOption={(input, option) => {
        const repo = validRepositories.find(r => r.id === option.value);
        if (!repo) return false;
        
        const searchText = input.toLowerCase();
        return (
          repo.displayName.toLowerCase().includes(searchText) ||
          repo.description.toLowerCase().includes(searchText) ||
          repo.ownerName.toLowerCase().includes(searchText) ||
          repo.name.toLowerCase().includes(searchText)
        );
      }}
      optionFilterProp="children"
      size="middle"
      allowClear
      notFoundContent={
        loading ? "Đang tải..." : "Không tìm thấy repository"
      }
      dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
    >
      {validRepositories.map(renderOption)}
    </Select>
  );
};

RepositorySelector.propTypes = {
  repositories: PropTypes.arrayOf(PropTypes.shape({
    owner: PropTypes.oneOfType([
      PropTypes.string,
      PropTypes.shape({
        login: PropTypes.string,
        name: PropTypes.string,
        avatar_url: PropTypes.string
      })
    ]).isRequired,
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    private: PropTypes.bool,
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
  })),
  loading: PropTypes.bool,
  selectedRepo: PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    owner: PropTypes.oneOfType([PropTypes.string, PropTypes.object]),
    name: PropTypes.string,
    key: PropTypes.string,
    displayName: PropTypes.string
  }),
  onRepositoryChange: PropTypes.func,
  placeholder: PropTypes.string
};

export default RepositorySelector;
