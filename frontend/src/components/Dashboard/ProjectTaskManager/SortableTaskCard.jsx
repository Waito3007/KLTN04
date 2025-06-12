// SortableTaskCard.jsx
import React from 'react';
import { Card, Avatar, Tag, Space, Typography, Button, Tooltip } from 'antd';
import { EditOutlined, DeleteOutlined, UserOutlined, CalendarOutlined } from '@ant-design/icons';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { formatDate, formatFullDate } from './kanbanUtils';
import { TASK_CARD_CONFIG } from './kanbanConstants';
import styles from './KanbanBoard.module.css';

const { Text } = Typography;

const SortableTaskCard = ({ 
  task, 
  getAssigneeInfo, 
  getPriorityColor, 
  showTaskModal, 
  deleteTask 
}) => {  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ 
    id: task.id,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition: transition || 'transform 200ms cubic-bezier(0.18, 0.67, 0.6, 1.22)',
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 999 : 'auto',
    cursor: isDragging ? 'grabbing' : 'grab',
    touchAction: 'none',
  };

  const assigneeInfo = getAssigneeInfo(task.assignee);

  return (    <Card
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={`${styles.taskCard} ${isDragging ? styles.isDragging : ''}`}
      styles={{ body: { padding: 12 } }}
    >
      <div className={styles.taskCardHeader}>
        <div className={styles.taskTitle}>{task.title}</div>
        <div className={styles.taskActions}>
          <Tooltip title="Chỉnh sửa">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                showTaskModal(task);
              }}
            />
          </Tooltip>
          <Tooltip title="Xóa">
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                deleteTask(task.id);
              }}
            />
          </Tooltip>
        </div>
      </div>

      {task.description && (
        <Text className={styles.taskDescription}>
          {task.description}
        </Text>
      )}

      <div className={styles.taskMeta}>
        <Space size="small">
          <Tag color={getPriorityColor(task.priority)} style={{ margin: 0 }}>
            {task.priority?.toUpperCase()}
          </Tag>
          {task.due_date && (
            <Tooltip title={`Hạn: ${formatFullDate(task.due_date)}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#666' }}>
                <CalendarOutlined style={{ fontSize: 12 }} />
                <Text style={{ fontSize: 11, color: '#666' }}>
                  {formatDate(task.due_date)}
                </Text>
              </div>
            </Tooltip>
          )}
        </Space>
      </div>

      <div className={styles.taskFooter}>
        <Space>
          <Avatar 
            size={TASK_CARD_CONFIG.AVATAR_SIZE} 
            src={assigneeInfo.avatar_url} 
            icon={<UserOutlined />}
          />
          <Text style={{ fontSize: 12, color: '#666' }}>
            {assigneeInfo.login}
          </Text>
        </Space>
        <Text style={{ fontSize: 11, color: '#999' }}>
          #{task.id}
        </Text>
      </div>
    </Card>
  );
};

export default SortableTaskCard;
