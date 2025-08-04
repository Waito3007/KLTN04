/**
 * TaskAssignBoard - Component chính cho việc phân công task
 * Tuân thủ quy tắc KLTN04: Tách biệt logic, sử dụng hooks, defensive programming
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, Row, Col, Alert, Spin, Button, Space, Typography } from 'antd';
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';
import RepositorySelector from './RepositorySelector';
import KanbanBoard from './KanbanBoard';
import CreateTaskModal from './CreateTaskModal';
import TaskStatsCard from './TaskStatsCard';
import useTaskAssign from '../../../hooks/useTaskAssign';
import { getTaskStats } from '../../../utils/taskUtils';
import './TaskAssignBoard.css';

const { Title, Text } = Typography;

const TaskAssignBoard = ({ repositories = [], repoLoading = false }) => {
  // State management - tuân thủ immutability
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);

  // Custom hooks cho logic nghiệp vụ
  const { 
    tasks, 
    loading, 
    error, 
    loadTasks, 
    createTask, 
    updateTask, 
    deleteTask,
    updateTaskStatus 
  } = useTaskAssign(selectedRepo);

  // Computed stats using utility function
  const taskStats = useMemo(() => 
    tasks.length > 0 ? getTaskStats(tasks) : null, 
    [tasks]
  );

  // Sử dụng repositories từ props thay vì hook
  // const { repositories, loading: repoLoading } = useRepositories();

  // Defensive programming: Validate dữ liệu
  const validTasks = useMemo(() => {
    if (!Array.isArray(tasks)) return [];
    return tasks.filter(task => task && typeof task === 'object' && task.id);
  }, [tasks]);

  // Effect để fetch tasks khi repository thay đổi
  useEffect(() => {
    const ownerName = typeof selectedRepo?.owner === 'string' 
      ? selectedRepo.owner 
      : selectedRepo?.owner?.login || selectedRepo?.owner?.name;
    
    if (ownerName && selectedRepo?.name) {
      loadTasks();
    }
  }, [selectedRepo, loadTasks]);

  // Handlers với error handling
  const handleRepositoryChange = (repo) => {
    try {
      setSelectedRepo(repo);
      // Không cần refreshTrigger, useEffect sẽ tự động load tasks
    } catch (error) {
      console.error('Lỗi khi thay đổi repository:', error);
    }
  };

  const handleCreateTask = async (taskData) => {
    try {
      if (!selectedRepo) {
        throw new Error('Vui lòng chọn repository trước');
      }

      const newTaskData = {
        ...taskData,
        repo_owner: typeof selectedRepo.owner === 'string' 
          ? selectedRepo.owner 
          : selectedRepo.owner?.login || selectedRepo.owner?.name,
        repo_name: selectedRepo.name
      };

      await createTask(newTaskData);
      setIsCreateModalVisible(false);
      loadTasks(); // Reload tasks sau khi tạo thành công
    } catch (error) {
      console.error('Lỗi khi tạo task:', error);
      // Error sẽ được handle bởi useTaskAssign hook
    }
  };

  const handleTaskUpdate = async (taskId, updateData) => {
    try {
      await updateTask(taskId, updateData);
      loadTasks(); // Reload tasks sau khi update
    } catch (error) {
      console.error('Lỗi khi cập nhật task:', error);
    }
  };

  const handleTaskDelete = async (taskId) => {
    try {
      await deleteTask(taskId);
      loadTasks(); // Reload tasks sau khi xóa
    } catch (error) {
      console.error('Lỗi khi xóa task:', error);
    }
  };

  const handleStatusChange = async (taskId, newStatus) => {
    try {
      await updateTaskStatus(taskId, newStatus);
      loadTasks(); // Reload tasks sau khi thay đổi status
    } catch (error) {
      console.error('Lỗi khi thay đổi status:', error);
    }
  };

  const handleRefresh = () => {
    loadTasks(); // Sử dụng loadTasks thay vì setRefreshTrigger
  };

  // Render error state
  if (error) {
    return (
      <Card className="task-assign-board">
        <Alert
          message="Lỗi khi tải dữ liệu"
          description={error.message || 'Có lỗi xảy ra khi tải dữ liệu task'}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={handleRefresh}>
              Thử lại
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <div className="task-assign-board">
      {/* Header với repository selector */}
      <Card className="task-assign-header" size="small">
        <Row align="middle" justify="space-between">
          <Col xs={24} sm={12} md={8}>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Text strong>Chọn Repository:</Text>
              <RepositorySelector
                repositories={repositories}
                loading={repoLoading}
                selectedRepo={selectedRepo}
                onRepositoryChange={handleRepositoryChange}
                placeholder="Chọn repository để quản lý tasks"
              />
            </Space>
          </Col>
          
          <Col xs={24} sm={12} md={8} style={{ textAlign: 'right' }}>
            {selectedRepo && (
              <Space>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setIsCreateModalVisible(true)}
                  disabled={loading}
                >
                  Tạo Task
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={loading}
                >
                  Làm mới
                </Button>
              </Space>
            )}
          </Col>
        </Row>
      </Card>

      {/* Task Statistics */}
      {selectedRepo && taskStats && (
        <TaskStatsCard stats={taskStats} loading={loading} />
      )}

      {/* Main Content */}
      {!selectedRepo ? (
        <Card className="task-assign-placeholder">
          <div style={{ textAlign: 'center', padding: '40px 20px' }}>
            <Title level={4} type="secondary">
              Vui lòng chọn repository để bắt đầu quản lý tasks
            </Title>
            <Text type="secondary">
              Chọn một repository từ danh sách trên để xem và quản lý các tasks được phân công
            </Text>
          </div>
        </Card>
      ) : (
        <Spin spinning={loading} tip="Đang tải tasks...">
          <KanbanBoard
            tasks={validTasks}
            onTaskUpdate={handleTaskUpdate}
            onTaskDelete={handleTaskDelete}
            onStatusChange={handleStatusChange}
            selectedRepo={selectedRepo}
          />
        </Spin>
      )}

      {/* Create Task Modal */}
      <CreateTaskModal
        visible={isCreateModalVisible}
        onCancel={() => setIsCreateModalVisible(false)}
        onSubmit={handleCreateTask}
        selectedRepo={selectedRepo}
        loading={loading}
      />
    </div>
  );
};

export default TaskAssignBoard;
