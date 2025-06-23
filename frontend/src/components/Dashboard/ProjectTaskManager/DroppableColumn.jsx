// DroppableColumn.jsx
import React from 'react';
import { useDroppable } from '@dnd-kit/core';
import styles from './KanbanBoard.module.css';

const DroppableColumn = ({ columnId, children }) => {
  const { setNodeRef, isOver } = useDroppable({
    id: columnId,
  });

  return (
    <div 
      ref={setNodeRef} 
      className={`${styles.tasksContainer} ${isOver ? styles.dragOver : ''}`}
    >
      {children}
    </div>
  );
};

export default DroppableColumn;
