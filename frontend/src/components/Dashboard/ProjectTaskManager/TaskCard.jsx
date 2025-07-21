import React, { useState } from 'react';
import { Card, Avatar, Tag, Space, Typography, Button, Tooltip } from 'antd';
import { EditOutlined, DeleteOutlined, UserOutlined, CalendarOutlined, LeftOutlined, RightOutlined } from '@ant-design/icons';
import { formatDate, formatFullDate } from './kanbanUtils';
import { getAvatarUrl } from '../../../utils/taskUtils.jsx';
import { TASK_CARD_CONFIG } from './kanbanConstants';
import styles from './KanbanBoard.module.css';

const { Text } = Typography;

const TaskCard = ({
  task,
  getAssigneeInfo,
  getPriorityColor,
  showTaskModal,
  deleteTask,
  onStatusChange // New prop for status change
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const assigneeInfo = getAssigneeInfo(task.assignee);

  return (
    <Card
      className={styles.taskCard}
      styles={{ body: { padding: 12 } }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
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
                <Text style={{ fontSize: 11, color: '#666'}}>
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
            src={getAvatarUrl(assigneeInfo.avatar_url, assigneeInfo.login)}
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

      {isHovered && (
        <div className={styles.statusChangeArrows}>
          <Button
            type="text"
            icon={<LeftOutlined />}
            onClick={() => onStatusChange(task.id, task.status, 'left')}
            className={styles.arrowButtonLeft}
          />
          <Button
            type="text"
            icon={<RightOutlined />}
            onClick={() => onStatusChange(task.id, task.status, 'right')}
            className={styles.arrowButtonRight}
          />
        </div>
      )}
    </Card>
  );
};

export default TaskCard;