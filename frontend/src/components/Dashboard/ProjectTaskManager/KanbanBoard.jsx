// KanbanBoard.jsx
import React from 'react';
import { Typography } from 'antd';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';

// Import custom modules
import { COLUMN_CONFIG } from './kanbanConstants';
import { getTasksByStatus } from './kanbanUtils';
import TaskCard from './TaskCard'; // Sẽ tạo hoặc sửa đổi TaskCard.jsx

// Import CSS Module
import styles from './KanbanBoard.module.css';

const { Title } = Typography;

const KanbanBoard = ({
  tasks = [],
  getAssigneeInfo,
  getPriorityColor,
  showTaskModal,
  deleteTask,
  updateTaskStatus
}) => {
  const safeTasks = Array.isArray(tasks) ? tasks : [];

  const handleStatusChange = (taskId, currentStatus, direction) => {
    const currentColumnIndex = COLUMN_CONFIG.findIndex(col => col.id === currentStatus);
    let newStatus = currentStatus;

    if (direction === 'left') {
      if (currentColumnIndex > 0) {
        newStatus = COLUMN_CONFIG[currentColumnIndex - 1].id;
      }
    } else if (direction === 'right') {
      if (currentColumnIndex < COLUMN_CONFIG.length - 1) {
        newStatus = COLUMN_CONFIG[currentColumnIndex + 1].id;
      }
    }

    if (newStatus !== currentStatus) {
      // Cập nhật UI ngay lập tức
      updateTaskStatus(taskId, newStatus, { optimistic: true });
    }
  };

  return (
    <div className={styles.kanbanContainer}>
      {COLUMN_CONFIG.map(column => {
        const columnTasks = getTasksByStatus(safeTasks, column.id);
        const IconComponent = column.icon;

        return (
          <div key={column.id} className={styles.kanbanColumn}>
            <div className={`${styles.columnHeader} ${styles[column.cssClass + 'Border']}`}>
              <Title
                level={5}
                className={`${styles.columnTitle} ${styles[column.cssClass + 'Color']}`}
              >
                <IconComponent />
                {column.title}
              </Title>
              <div className={`${styles.taskCount} ${styles[column.cssClass + 'Bg']}`}>
                {columnTasks.length}
              </div>
            </div>

            <div className={styles.columnContent}> {/* Thay thế DroppableColumn */}
              {columnTasks.map(task => (
                <TaskCard
                  key={task.id}
                  task={task}
                  getAssigneeInfo={getAssigneeInfo}
                  getPriorityColor={getPriorityColor}
                  showTaskModal={showTaskModal}
                  deleteTask={deleteTask}
                  onStatusChange={handleStatusChange}
                />
              ))}

              {/* Empty state when no tasks */}
              {columnTasks.length === 0 && (
                <div className={styles.emptyState}>
                  <IconComponent />
                  <span>Không có task nào</span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default KanbanBoard;