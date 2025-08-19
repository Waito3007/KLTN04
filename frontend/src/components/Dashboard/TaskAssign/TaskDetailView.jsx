import React, { useEffect, useState } from 'react';
import { Space, Tag, Typography, Divider, Row, Col, Button } from 'antd';
import { UserOutlined, CalendarOutlined, ClockCircleOutlined, LinkOutlined } from '@ant-design/icons';
import axios from 'axios';
import TaskCommitLinker from './TaskCommitLinker';

const { Text, Title } = Typography;

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const TaskDetailView = ({ task, selectedRepo, formatDate, formatDueDate, getPriorityConfig, getStatusConfig }) => {
  const [commits, setCommits] = useState([]);
  const [showCommitLinker, setShowCommitLinker] = useState(false);

  useEffect(() => {
    if (task && selectedRepo) {
      const fetchCommits = async () => {
        try {
          const owner = selectedRepo.owner?.login || selectedRepo.owner;
          const repo = selectedRepo.name;
          
          // Lấy token từ localStorage
          const token = localStorage.getItem('access_token');
          
          const response = await axios.get(
            `${API_BASE_URL}/projects/${owner}/${repo}/tasks/${task.id}/commits`,
            {
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
              }
            }
          );
          setCommits(response.data);
        } catch (error) {
          console.error('Error fetching commits:', error);
        }
      };
      fetchCommits();
    }
  }, [task, selectedRepo]);

  if (!task) return null;

  const priorityConfig = getPriorityConfig(task.priority);
  const statusConfig = getStatusConfig(task.status);

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {/* Title */}
      <div>
        <Title level={4} style={{ margin: 0 }}>
          {task.title}
        </Title>
      </div>

      {/* Status and Priority */}
      <Space size="middle">
        <Tag color={statusConfig.color} style={{ fontSize: '14px', padding: '4px 12px' }}>
          {statusConfig.label}
        </Tag>
        <Tag color={priorityConfig.color} style={{ fontSize: '14px', padding: '4px 12px' }}>
          {priorityConfig.label}
        </Tag>
      </Space>

      {/* Description */}
      {task.description && (
        <div>
          <Text strong>Mô tả:</Text>
          <div style={{ marginTop: 8, padding: 12, backgroundColor: '#fafafa', borderRadius: 6 }}>
            <Text>{task.description}</Text>
          </div>
        </div>
      )}

      <Divider />

      {/* Task Details */}
      <Row gutter={[16, 8]}>
        <Col span={12}>
          <Space>
            <UserOutlined />
            <Text strong>Người thực hiện:</Text>
            <Text>{task.assignee_github_username || 'Chưa phân công'}</Text>
          </Space>
        </Col>
        <Col span={12}>
          <Space>
            <CalendarOutlined />
            <Text strong>Hạn cuối:</Text>
            <Text>{formatDueDate(task.due_date)}</Text>
          </Space>
        </Col>
        <Col span={12}>
          <Space>
            <ClockCircleOutlined />
            <Text strong>Tạo lúc:</Text>
            <Text>{formatDate(task.created_at)}</Text>
          </Space>
        </Col>
        <Col span={12}>
          <Space>
            <ClockCircleOutlined />
            <Text strong>Cập nhật:</Text>
            <Text>{formatDate(task.updated_at)}</Text>
          </Space>
        </Col>
        {task.created_by && (
          <Col span={24}>
            <Space>
              <UserOutlined />
              <Text strong>Tạo bởi:</Text>
              <Text>{task.created_by}</Text>
            </Space>
          </Col>
        )}
      </Row>

      {/* Commits Section */}
      {commits.length > 0 && (
        <div>
          <Text strong>Danh sách Commit đã liên kết:</Text>
          <ul>
            {commits.map((commit) => (
              <li key={commit.sha}>
                <a href={commit.url} target="_blank" rel="noopener noreferrer">
                  {commit.message} - {commit.author_name}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Manual Commit Link Button */}
      {task?.assignee_github_username && (
        <div style={{ marginTop: '16px', textAlign: 'center' }}>
          <Button 
            type="primary" 
            icon={<LinkOutlined />}
            onClick={() => setShowCommitLinker(true)}
          >
            Liên kết commit thủ công
          </Button>
        </div>
      )}

      {/* TaskCommitLinker Modal */}
      <TaskCommitLinker
        task={task}
        selectedRepo={selectedRepo}
        visible={showCommitLinker}
        onClose={() => setShowCommitLinker(false)}
        onCommitLinked={() => {
          // Refresh commits sau khi liên kết thành công
          if (task && selectedRepo) {
            const owner = selectedRepo.owner?.login || selectedRepo.owner;
            const repo = selectedRepo.name;
            
            // Lấy token từ localStorage
            const token = localStorage.getItem('access_token');
            
            axios.get(
              `${API_BASE_URL}/projects/${owner}/${repo}/tasks/${task.id}/commits`,
              {
                headers: {
                  'Authorization': `Bearer ${token}`,
                  'Content-Type': 'application/json'
                }
              }
            )
              .then(response => setCommits(response.data))
              .catch(error => console.error('Error refreshing commits:', error));
          }
        }}
      />
    </Space>
  );
};

export default TaskDetailView;
