/**
 * KanbanColumn - Cột trong bảng Kanban
 * Tuân thủ quy tắc KLTN04: Component nhỏ, single responsibility
 */

import React, { useMemo } from 'react';
import { Card, Badge, Space, Typography, Tooltip, Button } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import PropTypes from 'prop-types';
import TaskCard from './TaskCard';
import './KanbanColumn.css';

const { Title, Text } = Typography;

const KanbanColumn = ({ 
  status, 
  tasks = [], 
  onTaskClick, 
  onTaskMove 
}) => {
  // Defensive programming: Validate tasks
  const validTasks = useMemo(() => {
    if (!Array.isArray(tasks)) return [];
    return tasks.filter(task => task && task.id);
  }, [tasks]);

  // Tính toán thống kê
  const taskStats = useMemo(() => {
    const total = validTasks.length;
    const priorities = validTasks.reduce((acc, task) => {
      const priority = task.priority || 'MEDIUM';
      acc[priority] = (acc[priority] || 0) + 1;
      return acc;
    }, {});

    return { total, priorities };
  }, [validTasks]);

  // Handler để tạo task mới với status cụ thể
  const handleAddTask = () => {
    // Emit event để parent component handle
    // Có thể implement modal tạo task với status preset
    console.log(`Add new task with status: ${status.key}`);
  };

  // Get priority color
  const getPriorityColor = (priority) => {
    const colors = {
      LOW: '#52c41a',
      MEDIUM: '#1890ff', 
      HIGH: '#fa8c16',
      URGENT: '#ff4d4f'
    };
    return colors[priority] || colors.MEDIUM;
  };

  return (
    <Card 
      className="kanban-column"
      size="small"
      style={{ 
        borderTop: `4px solid ${status.color}`,
        backgroundColor: status.bgColor 
      }}
    >
      {/* Column Header */}
      <div className="kanban-column-header">
        <Space justify="space-between" style={{ width: '100%' }}>
          <Space>
            <Title level={5} style={{ margin: 0, color: status.color }}>
              {status.title}
            </Title>
            <Badge 
              count={taskStats.total} 
              style={{ backgroundColor: status.color }}
            />
          </Space>
          
          <Tooltip title={`Thêm task ${status.title.toLowerCase()}`}>
            <Button 
              type="text" 
              size="small" 
              icon={<PlusOutlined />}
              onClick={handleAddTask}
              style={{ color: status.color }}
            />
          </Tooltip>
        </Space>
      </div>

      {/* Priority Stats */}
      {taskStats.total > 0 && (
        <div className="kanban-column-stats">
          <Space size="small">
            {Object.entries(taskStats.priorities).map(([priority, count]) => (
              <Text 
                key={priority}
                style={{ 
                  fontSize: '11px',
                  color: getPriorityColor(priority),
                  fontWeight: 500
                }}
              >
                {priority}: {count}
              </Text>
            ))}
          </Space>
        </div>
      )}

      {/* Tasks List */}
      <div className="kanban-column-content">
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          {validTasks.map(task => (
            <TaskCard
              key={task.id}
              task={task}
              onClick={() => onTaskClick?.(task)}
              onMove={onTaskMove}
              allowedStatuses={Object.keys(taskStats.total > 0 ? {} : {})} // Will be implemented
            />
          ))}
          
          {validTasks.length === 0 && (
            <div className="kanban-column-empty">
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Không có task nào
              </Text>
            </div>
          )}
        </Space>
      </div>
    </Card>
  );
};

export default KanbanColumn;
