/**
 * KanbanBoard - Bảng Kanban để quản lý tasks
 * Tuân thủ quy tắc KLTN04: Immutability, defensive programming
 */

import React, { useMemo, useState } from 'react';
import { Row, Col, Card, Empty, Button, Space, Typography } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import KanbanColumn from './KanbanColumn';
import TaskDetailModal from './TaskDetailModal';
import './KanbanBoard.css';

const { Title } = Typography;

// Constants cho task statuses - tuân thủ nguyên tắc không magic values
const TASK_STATUSES = {
  TODO: { 
    key: 'TODO', 
    title: 'Cần làm', 
    color: '#595959',
    bgColor: '#ffffff'
  },
  IN_PROGRESS: { 
    key: 'IN_PROGRESS', 
    title: 'Đang làm', 
    color: '#262626',
    bgColor: '#fafafa'
  },
  DONE: { 
    key: 'DONE', 
    title: 'Hoàn thành', 
    color: '#000000',
    bgColor: '#f5f5f5'
  }
};

const KanbanBoard = ({ 
  tasks = [], 
  onTaskUpdate, 
  onTaskDelete, 
  onStatusChange, 
  selectedRepo 
}) => {
  const [selectedTask, setSelectedTask] = useState(null);
  const [isDetailModalVisible, setIsDetailModalVisible] = useState(false);

  // Defensive programming: Group tasks by status với validation
  const tasksByStatus = useMemo(() => {
    const grouped = {};
    
    // Initialize tất cả statuses
    Object.values(TASK_STATUSES).forEach(status => {
      grouped[status.key] = [];
    });

    // Group tasks với validation
    if (Array.isArray(tasks)) {
      tasks.forEach(task => {
        if (task && task.status && grouped[task.status]) {
          grouped[task.status].push(task);
        } else if (task && !task.status) {
          // Default status nếu không có
          grouped.TODO.push({ ...task, status: 'TODO' });
        }
      });
    }

    return grouped;
  }, [tasks]);

  // Handlers với error handling
  const handleTaskClick = (task) => {
    try {
      setSelectedTask(task);
      setIsDetailModalVisible(true);
    } catch (error) {
      console.error('Lỗi khi mở task detail:', error);
    }
  };

  const handleTaskMove = async (taskId, newStatus) => {
    try {
      if (!taskId || !newStatus) {
        throw new Error('Task ID và status mới là bắt buộc');
      }
      
      await onStatusChange?.(taskId, newStatus);
    } catch (error) {
      console.error('Lỗi khi di chuyển task:', error);
      // Có thể thêm notification ở đây
    }
  };

  const handleTaskUpdate = async (taskId, updateData) => {
    try {
      await onTaskUpdate?.(taskId, updateData);
      setIsDetailModalVisible(false);
      setSelectedTask(null);
    } catch (error) {
      console.error('Lỗi khi cập nhật task:', error);
    }
  };

  const handleTaskDelete = async (taskId) => {
    try {
      await onTaskDelete?.(taskId);
      setIsDetailModalVisible(false);
      setSelectedTask(null);
    } catch (error) {
      console.error('Lỗi khi xóa task:', error);
    }
  };

  const closeDetailModal = () => {
    setIsDetailModalVisible(false);
    setSelectedTask(null);
  };

  // Render empty state
  if (!Array.isArray(tasks) || tasks.length === 0) {
    return (
      <Card className="kanban-board-empty">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <Space direction="vertical" size="small">
              <Typography.Text type="secondary">
                Chưa có task nào cho repository này
              </Typography.Text>
              {selectedRepo && (
                <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                  Repository: {typeof selectedRepo.owner === 'string' ? selectedRepo.owner : selectedRepo.owner?.login || 'unknown'}/{selectedRepo.name}
                </Typography.Text>
              )}
            </Space>
          }
        >
          <Button type="primary" icon={<PlusOutlined />}>
            Tạo task đầu tiên
          </Button>
        </Empty>
      </Card>
    );
  }

  return (
    <div className="kanban-board">
      <Row gutter={[16, 16]} className="kanban-columns">
        {Object.values(TASK_STATUSES).map(status => (
          <Col key={status.key} xs={24} sm={24} md={8} lg={8}>
            <KanbanColumn
              status={status}
              tasks={tasksByStatus[status.key] || []}
              onTaskClick={handleTaskClick}
              onTaskMove={handleTaskMove}
              selectedRepo={selectedRepo}
            />
          </Col>
        ))}
      </Row>

      {/* Task Detail Modal */}
      <TaskDetailModal
        visible={isDetailModalVisible}
        task={selectedTask}
        onCancel={closeDetailModal}
        onUpdate={handleTaskUpdate}
        onDelete={handleTaskDelete}
        selectedRepo={selectedRepo}
      />
    </div>
  );
};

export default KanbanBoard;
