/**
 * TaskCard - Card hiển thị thông tin task trong Kanban
 * Tuân thủ quy tắc KLTN04: Component nhỏ, reusable
 */

import React, { useMemo } from 'react';
import { Card, Tag, Space, Typography, Avatar, Tooltip, Button } from 'antd';
import { 
  CalendarOutlined, 
  UserOutlined, 
  EditOutlined,
  ClockCircleOutlined 
} from '@ant-design/icons';
import { format, isAfter, parseISO } from 'date-fns';
import { vi } from 'date-fns/locale';
import PropTypes from 'prop-types';
import './TaskCard.css';

const { Text, Paragraph } = Typography;

const TaskCard = ({ task, onClick }) => {
  // Defensive programming: Validate task data
  const validTask = useMemo(() => {
    if (!task || typeof task !== 'object') return null;
    
    return {
      id: task.id,
      title: task.title || 'Untitled Task',
      description: task.description || '',
      status: task.status || 'TODO',
      priority: task.priority || 'MEDIUM',
      assignee_github_username: task.assignee_github_username,
      due_date: task.due_date,
      created_by: task.created_by,
      created_at: task.created_at,
      updated_at: task.updated_at
    };
  }, [task]);

  // Get priority config
  const priorityConfig = useMemo(() => {
    const configs = {
      LOW: { color: '#8c8c8c', text: 'Thấp' },
      MEDIUM: { color: '#595959', text: 'Trung bình' },
      HIGH: { color: '#262626', text: 'Cao' },
      URGENT: { color: '#000000', text: 'Khẩn cấp' }
    };
    return configs[validTask?.priority] || configs.MEDIUM;
  }, [validTask?.priority]);

  // Check if task is overdue
  const isOverdue = useMemo(() => {
    if (!validTask?.due_date) return false;
    try {
      const dueDate = parseISO(validTask.due_date);
      return isAfter(new Date(), dueDate) && validTask.status !== 'DONE';
    } catch {
      return false;
    }
  }, [validTask?.due_date, validTask?.status]);

  // Format due date
  const formattedDueDate = useMemo(() => {
    if (!validTask?.due_date) return null;
    try {
      const dueDate = parseISO(validTask.due_date);
      return format(dueDate, 'dd/MM/yyyy', { locale: vi });
    } catch {
      return null;
    }
  }, [validTask?.due_date]);

  if (!validTask) return null;

  const handleCardClick = (e) => {
    e.stopPropagation();
    onClick?.(validTask);
  };

  return (
    <Card
      className={`task-card ${isOverdue ? 'task-card-overdue' : ''}`}
      size="small"
      hoverable
      onClick={handleCardClick}
      style={{ cursor: 'pointer' }}
      data-priority={validTask.priority}
      data-status={validTask.status}
      actions={[
        <Tooltip key="edit" title="Chỉnh sửa">
          <EditOutlined onClick={handleCardClick} />
        </Tooltip>
      ]}
    >
      {/* Task Title */}
      <div className="task-card-title">
        <Paragraph 
          ellipsis={{ rows: 2, tooltip: validTask.title }}
          style={{ margin: 0, fontWeight: 500 }}
        >
          {validTask.title}
        </Paragraph>
      </div>

      {/* Task Description */}
      {validTask.description && (
        <div className="task-card-description">
          <Paragraph
            ellipsis={{ rows: 2, tooltip: validTask.description }}
            type="secondary"
            style={{ margin: '8px 0', fontSize: '12px' }}
          >
            {validTask.description}
          </Paragraph>
        </div>
      )}

      {/* Task Metadata */}
      <div className="task-card-metadata" style={{ marginTop: '8px' }}>
        {/* Priority Tag */}
        <div style={{ marginBottom: '6px' }}>
          <Tag color={priorityConfig.color} style={{ margin: 0 }}>
            {priorityConfig.text}
          </Tag>
        </div>

        {/* Due Date */}
        {formattedDueDate && (
          <div className="task-due-date" style={{ 
            marginBottom: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            flexWrap: 'nowrap'
          }}>
            <CalendarOutlined 
              style={{ 
                color: isOverdue ? '#ff4d4f' : '#8c8c8c',
                fontSize: '12px',
                flexShrink: 0
              }} 
            />
            <Text 
              style={{ 
                fontSize: '11px',
                color: isOverdue ? '#ff4d4f' : '#8c8c8c',
                whiteSpace: 'nowrap'
              }}
            >
              {formattedDueDate}
            </Text>
            {isOverdue && (
              <Tag color="error" size="small" style={{ marginLeft: '4px' }}>
                Quá hạn
              </Tag>
            )}
          </div>
        )}

        {/* Assignee */}
        {validTask.assignee_github_username && (
          <div className="task-assignee" style={{ 
            marginBottom: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            flexWrap: 'nowrap'
          }}>
            <Avatar 
              size="small" 
              icon={<UserOutlined />}
              style={{ backgroundColor: '#595959', flexShrink: 0 }}
            />
            <Text style={{ 
              fontSize: '11px', 
              color: '#8c8c8c',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              @{validTask.assignee_github_username}
            </Text>
          </div>
        )}

        {/* Created by */}
        {validTask.created_by && (
          <div className="task-created-by" style={{ 
            marginBottom: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            flexWrap: 'nowrap'
          }}>
            <ClockCircleOutlined style={{ 
              fontSize: '11px', 
              color: '#8c8c8c',
              flexShrink: 0
            }} />
            <Text style={{ 
              fontSize: '10px', 
              color: '#8c8c8c',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              Tạo bởi {validTask.created_by}
            </Text>
          </div>
        )}
      </div>
    </Card>
  );
};

// PropTypes
TaskCard.propTypes = {
  task: PropTypes.object.isRequired,
  onClick: PropTypes.func
};

export default TaskCard;
