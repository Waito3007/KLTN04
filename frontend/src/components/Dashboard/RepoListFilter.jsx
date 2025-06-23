import React, { useState } from 'react';
import { Card, Row, Col, Input, Select, Button } from 'antd';
import { SearchOutlined } from '@ant-design/icons';

const { Option } = Select;

const RepoListFilter = ({ onFilterChange }) => {
  const [searchText, setSearchText] = useState('');
  const [status, setStatus] = useState('all');
  const [assignee, setAssignee] = useState('all');

  const handleApplyFilter = () => {
    onFilterChange({ searchText, status, assignee });
  };

  return (
    <Card title="Bộ lọc Repository" variant="borderless">
      <Row gutter={16}>
        <Col span={8}>
          <Input
            placeholder="Tìm kiếm repo"
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
        </Col>
        <Col span={6}>
          <Select
            style={{ width: '100%' }}
            value={status}
            onChange={(value) => setStatus(value)}
            placeholder="Trạng thái"
          >
            <Option value="all">Tất cả</Option>
            <Option value="active">Đang hoạt động</Option>
            <Option value="archived">Đã lưu trữ</Option>
          </Select>
        </Col>
        <Col span={6}>
          <Select
            style={{ width: '100%' }}
            value={assignee}
            onChange={(value) => setAssignee(value)}
            placeholder="Người phụ trách"
          >
            <Option value="all">Tất cả</Option>
            <Option value="user1">User 1</Option>
            <Option value="user2">User 2</Option>
          </Select>
        </Col>
        <Col span={4}>
          <Button type="primary" onClick={handleApplyFilter}>
            Áp dụng
          </Button>
        </Col>
      </Row>
    </Card>
  );
};

export default RepoListFilter;