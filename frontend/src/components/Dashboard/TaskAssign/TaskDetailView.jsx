import React, { useEffect, useState } from 'react';
import { Space, Tag, Typography, Divider, Row, Col, Button, List, Card, Tooltip } from 'antd';
import { 
  UserOutlined, 
  CalendarOutlined, 
  ClockCircleOutlined, 
  LinkOutlined, 
  CodeOutlined, 
  FileTextOutlined, 
  PlusOutlined, 
  MinusOutlined,
  BranchesOutlined,
  ExportOutlined
} from '@ant-design/icons';
import axios from 'axios';
import TaskCommitLinker from './TaskCommitLinker';

const { Text, Title } = Typography;

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const TaskDetailView = ({ task, selectedRepo, formatDate, formatDueDate, getPriorityConfig, getStatusConfig }) => {
  const [commits, setCommits] = useState([]);
  const [showCommitLinker, setShowCommitLinker] = useState(false);
  const [commitsLoading, setCommitsLoading] = useState(false);

  const fetchCommits = async () => {
    if (!task || !selectedRepo) return;
    
    setCommitsLoading(true);
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
      setCommits(response.data || []);
    } catch (error) {
      console.error('Error fetching commits:', error);
      setCommits([]);
    } finally {
      setCommitsLoading(false);
    }
  };

  useEffect(() => {
    fetchCommits();
  }, [task, selectedRepo]); // eslint-disable-line react-hooks/exhaustive-deps

  const formatCommitDate = (dateString) => {
    if (!dateString) return 'Không có';
    try {
      return new Date(dateString).toLocaleString('vi-VN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'Không hợp lệ';
    }
  };

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

      {/* Enhanced Commits Section */}
      {commits.length > 0 && (
        <Card 
          size="small" 
          title={
            <Space>
              <BranchesOutlined />
              <Text strong>Danh sách Commit đã liên kết ({commits.length})</Text>
            </Space>
          }
          style={{ marginTop: 16 }}
          loading={commitsLoading}
        >
          <List
            itemLayout="vertical"
            dataSource={commits}
            renderItem={(commit) => (
              <List.Item
                key={commit.sha}
                actions={[
                  <Tooltip title="Xem commit trên GitHub">
                    <Button 
                      type="link" 
                      icon={<ExportOutlined />} 
                      size="small"
                      onClick={() => window.open(commit.url, '_blank')}
                    >
                      GitHub
                    </Button>
                  </Tooltip>
                ]}
              >
                <List.Item.Meta
                  title={
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <Text strong style={{ fontSize: '14px' }}>
                        <CodeOutlined style={{ marginRight: 8 }} />
                        {commit.message}
                      </Text>
                      <Space wrap>
                        <Tag color="blue" style={{ fontSize: '11px' }}>
                          {commit.sha?.substring(0, 8)}
                        </Tag>
                        <Tag color="green" style={{ fontSize: '11px' }}>
                          <UserOutlined style={{ marginRight: 4 }} />
                          {commit.author_name}
                        </Tag>
                        <Tag color="orange" style={{ fontSize: '11px' }}>
                          <CalendarOutlined style={{ marginRight: 4 }} />
                          {formatCommitDate(commit.committed_date)}
                        </Tag>
                      </Space>
                    </Space>
                  }
                  description={
                    <Space style={{ marginTop: 8 }}>
                      <Tooltip title="Dòng được thêm">
                        <Tag color="success" style={{ fontSize: '11px' }}>
                          <PlusOutlined style={{ marginRight: 2 }} />
                          {commit.insertions || 0}
                        </Tag>
                      </Tooltip>
                      <Tooltip title="Dòng được xóa">
                        <Tag color="error" style={{ fontSize: '11px' }}>
                          <MinusOutlined style={{ marginRight: 2 }} />
                          {commit.deletions || 0}
                        </Tag>
                      </Tooltip>
                      <Tooltip title="Số file thay đổi">
                        <Tag color="processing" style={{ fontSize: '11px' }}>
                          <FileTextOutlined style={{ marginRight: 2 }} />
                          {commit.files_changed || 0} files
                        </Tag>
                      </Tooltip>
                      <Text type="secondary" style={{ fontSize: '11px', marginLeft: 8 }}>
                        {commit.author_email}
                      </Text>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
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
        onCommitLinked={fetchCommits}
      />
    </Space>
  );
};

export default TaskDetailView;
