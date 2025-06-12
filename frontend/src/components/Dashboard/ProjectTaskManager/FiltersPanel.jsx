import React from 'react';
import { Row, Col, Select, Button, Space, Badge, Input, Avatar } from 'antd';
import { ReloadOutlined, FilterOutlined, SearchOutlined } from '@ant-design/icons';

const { Option } = Select;
const { Search } = Input;

const FiltersPanel = ({
  searchText,
  setSearchText,
  statusFilter,
  setStatusFilter,
  priorityFilter,
  setPriorityFilter,
  assigneeFilter,
  setAssigneeFilter,
  collaborators,
  fetchTasks,
  tasksLoading,
  filteredTasks
}) => (
  <Row gutter={16} align="middle">
    <Col span={6}>
      <Search
        placeholder="Tìm kiếm tasks..."
        value={searchText}
        onChange={(e) => setSearchText(e.target.value)}
        prefix={<SearchOutlined />}
        allowClear
      />
    </Col>
    <Col span={4}>
      <Select 
        placeholder="Trạng thái"
        value={statusFilter}
        onChange={setStatusFilter}
        style={{ width: '100%' }}
      >
        <Option value="all">Tất cả</Option>
        <Option value="todo">Chưa bắt đầu</Option>
        <Option value="in_progress">Đang làm</Option>
        <Option value="done">Hoàn thành</Option>
      </Select>
    </Col>
    <Col span={4}>
      <Select 
        placeholder="Độ ưu tiên"
        value={priorityFilter}
        onChange={setPriorityFilter}
        style={{ width: '100%' }}
      >
        <Option value="all">Tất cả</Option>
        <Option value="high">Cao</Option>
        <Option value="medium">Trung bình</Option>
        <Option value="low">Thấp</Option>
      </Select>
    </Col>
    <Col span={4}>
      <Select 
        placeholder="Người thực hiện"
        value={assigneeFilter}
        onChange={setAssigneeFilter}
        style={{ width: '100%' }}
      >
        <Option value="all">Tất cả</Option>
        {collaborators.map(collab => (
          <Option key={collab.login} value={collab.login}>
            <Space>
              <Avatar src={collab.avatar_url} size="small" />
              {collab.login}
            </Space>
          </Option>
        ))}
      </Select>
    </Col>
    <Col span={6}>
      <Space>
        <Button 
          icon={<ReloadOutlined />}
          onClick={fetchTasks}
          disabled={tasksLoading}
        >
          Làm mới
        </Button>
        <Badge count={filteredTasks.length} showZero>
          <Button icon={<FilterOutlined />}>
            Kết quả lọc
          </Button>
        </Badge>
      </Space>
    </Col>
  </Row>
);

export default FiltersPanel;
