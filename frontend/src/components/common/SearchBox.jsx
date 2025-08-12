// Search component tái sử dụng với các tính năng phong phú
import React, { useState, useEffect } from 'react';
import { Input, Select, Button, Space, Dropdown, Menu, Tag } from 'antd';
import { 
  SearchOutlined, 
  FilterOutlined, 
  ClearOutlined, 
  DownOutlined,
  CloseOutlined 
} from '@ant-design/icons';

const { Option } = Select;

const SearchBox = ({
  placeholder = 'Tìm kiếm...',
  onSearch,
  onFilter,
  searchValue = '',
  filters = [], // [{ key, label, options: [{ value, label }] }]
  activeFilters = {},
  size = 'default',
  style = {},
  debounceMs = 300,
  clearable = true,
  allowClear = true
}) => {
  const [searchText, setSearchText] = useState(searchValue);
  const [localFilters, setLocalFilters] = useState(activeFilters);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (onSearch && searchText !== searchValue) {
        onSearch(searchText);
      }
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [searchText, debounceMs, onSearch, searchValue]);

  const handleFilterChange = (filterKey, value) => {
    const newFilters = { ...localFilters };
    if (value === undefined || value === null || value === '') {
      delete newFilters[filterKey];
    } else {
      newFilters[filterKey] = value;
    }
    setLocalFilters(newFilters);
    if (onFilter) {
      onFilter(newFilters);
    }
  };

  const handleClearAll = () => {
    setSearchText('');
    setLocalFilters({});
    if (onSearch) onSearch('');
    if (onFilter) onFilter({});
  };

  const hasActiveFilters = Object.keys(localFilters).length > 0 || searchText;

  const renderFilterTag = (filterKey, value) => {
    const filter = filters.find(f => f.key === filterKey);
    if (!filter) return null;

    const option = filter.options?.find(opt => opt.value === value);
    const label = option ? option.label : value;

    return (
      <Tag
        key={filterKey}
        closable
        onClose={() => handleFilterChange(filterKey, undefined)}
        style={{
          borderRadius: '6px',
          border: '1px solid #d9d9d9',
          background: '#fafafa'
        }}
      >
        {filter.label}: {label}
      </Tag>
    );
  };

  const filterMenuItems = filters.map(filter => ({
    key: filter.key,
    label: (
      <div style={{ padding: '8px 12px' }}>
        <div style={{ marginBottom: '4px', fontSize: '12px', color: '#8c8c8c' }}>
          {filter.label}
        </div>
        <Select
          placeholder={`Chọn ${filter.label.toLowerCase()}`}
          style={{ width: '100%' }}
          value={localFilters[filter.key]}
          onChange={(value) => handleFilterChange(filter.key, value)}
          allowClear
          size="small"
          onClick={(e) => e.stopPropagation()} // Prevent dropdown from closing
        >
          {filter.options?.map(option => (
            <Option key={option.value} value={option.value}>
              {option.label}
            </Option>
          ))}
        </Select>
      </div>
    )
  }));

  return (
    <div style={style}>
      {/* Main search bar */}
      <Space.Compact style={{ width: '100%' }}>
        <Input
          placeholder={placeholder}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          prefix={<SearchOutlined style={{ color: '#8c8c8c' }} />}
          size={size}
          allowClear={allowClear}
          style={{
            borderRadius: '8px 0 0 8px',
            borderRight: 'none'
          }}
        />
        
        {filters.length > 0 && (
          <Dropdown 
            menu={{ 
              items: filterMenuItems,
              style: {
                borderRadius: '8px',
                padding: '8px',
                minWidth: '200px'
              }
            }}
            trigger={['click']}
            placement="bottomRight"
          >
            <Button
              icon={<FilterOutlined />}
              size={size}
              style={{
                borderRadius: '0',
                borderLeft: 'none',
                borderRight: 'none'
              }}
            >
              Lọc <DownOutlined />
            </Button>
          </Dropdown>
        )}

        {clearable && hasActiveFilters && (
          <Button
            icon={<ClearOutlined />}
            onClick={handleClearAll}
            size={size}
            style={{
              borderRadius: filters.length > 0 ? '0 8px 8px 0' : '0 8px 8px 0',
              borderLeft: 'none'
            }}
            title="Xóa tất cả"
          />
        )}
      </Space.Compact>

      {/* Active filters */}
      {hasActiveFilters && (
        <div style={{ 
          marginTop: '12px', 
          display: 'flex', 
          flexWrap: 'wrap', 
          gap: '8px',
          alignItems: 'center'
        }}>
          <span style={{ fontSize: '12px', color: '#8c8c8c' }}>
            Bộ lọc đang áp dụng:
          </span>
          
          {searchText && (
            <Tag
              closable
              onClose={() => setSearchText('')}
              style={{
                borderRadius: '6px',
                border: '1px solid #1890ff',
                background: '#e6f7ff',
                color: '#1890ff'
              }}
            >
              Tìm kiếm: "{searchText}"
            </Tag>
          )}
          
          {Object.entries(localFilters).map(([key, value]) => 
            renderFilterTag(key, value)
          )}
        </div>
      )}
    </div>
  );
};

export default SearchBox;
