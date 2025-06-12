// DragOverlayContent.jsx
import React from 'react';
import { Tag } from 'antd';
import { truncateDescription } from './kanbanUtils';
import styles from './KanbanBoard.module.css';

const DragOverlayContent = ({ activeTask, getPriorityColor }) => {
  if (!activeTask) return null;

  return (
    <div className={styles.dragOverlay}>
      <div className={styles.dragOverlayContent}>
        <div className={styles.dragOverlayTitle}>
          {activeTask.title}
        </div>
        {activeTask.description && (
          <div className={styles.dragOverlayDescription}>
            {truncateDescription(activeTask.description)}
          </div>
        )}
        <div className={styles.dragOverlayFooter}>
          <Tag 
            color={getPriorityColor(activeTask.priority)} 
            className={styles.dragOverlayTag}
          >
            {activeTask.priority?.toUpperCase()}
          </Tag>
          <span className={styles.dragOverlayId}>
            #{activeTask.id}
          </span>
        </div>
      </div>
    </div>
  );
};

export default DragOverlayContent;
