import React from 'react';
import { Card, Space, Button, Tooltip, Tag, Avatar, Select } from 'antd';
import { EditOutlined, DeleteOutlined, UserOutlined, CalendarOutlined } from '@ant-design/icons';
import { getAvatarUrl } from '../../../utils/taskUtils.jsx';
import styled from 'styled-components';

const { Option } = Select;

const TaskHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
`;

const TaskActions = styled.div`
  display: flex;
  gap: 8px;
`;

const PriorityTag = styled(Tag)`
  font-weight: 500;
`;

const TaskCard = ({
  task,
  getAssigneeInfo,
  getStatusIcon,
  getPriorityColor,
  updateTaskStatus,
  showTaskModal,
  deleteTask
}) => {
  const assigneeInfo = getAssigneeInfo(task.assignee);
  return (
    <Card size="small">
      <TaskHeader>
        <Space>
          {getStatusIcon(task.status)}
          <strong>{task.title}</strong>
          <PriorityTag color={getPriorityColor(task.priority)}>
            {task.priority?.toUpperCase()}
          </PriorityTag>
        </Space>
        <TaskActions>
          <Tooltip title="Chỉnh sửa">
            <Button 
              size="small" 
              icon={<EditOutlined />}
              onClick={() => showTaskModal(task)}
            />
          </Tooltip>
          <Tooltip title="Xóa">
            <Button 
              size="small" 
              danger
              icon={<DeleteOutlined />}
              onClick={() => deleteTask(task.id)}
            />
          </Tooltip>
        </TaskActions>
      </TaskHeader>
      <div style={{ marginBottom: 8 }}>
        {task.description}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>          <Avatar 
            src={getAvatarUrl(assigneeInfo.avatar_url, assigneeInfo.login || assigneeInfo.github_username)} 
            icon={<UserOutlined />}
            size="small"
          /><div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: 12, fontWeight: 500 }}>
              {assigneeInfo.display_name || assigneeInfo.login || assigneeInfo.github_username}
            </span>
            {(assigneeInfo.type || assigneeInfo.is_owner || assigneeInfo.role) && (
              <Tag 
                size="small" 
                color={
                  assigneeInfo.is_owner ? 'gold' : 
                  assigneeInfo.type === 'Owner' ? 'gold' : 
                  'blue'
                }
                style={{ fontSize: '9px', marginTop: 2 }}
              >
                {assigneeInfo.is_owner ? 'Owner' : assigneeInfo.type || assigneeInfo.role}
              </Tag>
            )}
          </div>
        </Space>
        <Space>
          {task.due_date && (
            <Space style={{ fontSize: 12, color: '#666' }}>
              <CalendarOutlined />
              {task.due_date}
            </Space>
          )}
          <Select 
            size="small"
            value={task.status}
            onChange={(newStatus) => updateTaskStatus(task.id, newStatus)}
            style={{ width: 100 }}
          >
            <Option value="todo">To Do</Option>
            <Option value="in_progress">Đang làm</Option>
            <Option value="done">Hoàn thành</Option>
          </Select>
        </Space>
      </div>
    </Card>
  );
};

export default TaskCard;
