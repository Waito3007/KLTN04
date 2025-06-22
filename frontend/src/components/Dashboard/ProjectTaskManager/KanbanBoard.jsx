// KanbanBoard.jsx
import React from 'react';
import { Typography } from 'antd';
import { DndContext, closestCenter, DragOverlay } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';

// Import custom modules
import { COLUMN_CONFIG } from './kanbanConstants';
import { useKanbanDragDrop } from './useKanbanDragDrop';
import { getTasksByStatus } from './kanbanUtils';
import DroppableColumn from './DroppableColumn';
import SortableTaskCard from './SortableTaskCard';
import DragOverlayContent from './DragOverlayContent';

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
  // Safe array check
  const safeTasks = Array.isArray(tasks) ? tasks : [];

  // Use custom hook for drag & drop logic
  const {
    sensors,
    activeTask,
    handleDragStart,
    handleDragEnd,
    dropAnimation
  } = useKanbanDragDrop({ 
    tasks: safeTasks, 
    updateTaskStatus, 
    columns: COLUMN_CONFIG 
  });  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
    >
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

              <DroppableColumn columnId={column.id}>
                <SortableContext 
                  items={columnTasks.map(task => task.id)}
                  strategy={verticalListSortingStrategy}
                >
                  {columnTasks.map(task => (
                    <SortableTaskCard
                      key={task.id}
                      task={task}
                      getAssigneeInfo={getAssigneeInfo}
                      getPriorityColor={getPriorityColor}
                      showTaskModal={showTaskModal}
                      deleteTask={deleteTask}
                    />
                  ))}
                </SortableContext>
                
                {/* Empty state when no tasks */}
                {columnTasks.length === 0 && (
                  <div className={styles.emptyState}>
                    <IconComponent />
                    <span>Kéo task vào đây</span>
                  </div>
                )}
              </DroppableColumn>
            </div>
          );
        })}
      </div>      {/* Drag Overlay */}
      <DragOverlay 
        adjustScale={false}
        dropAnimation={dropAnimation}
        modifiers={[]}
        style={{
          cursor: 'grabbing',
          zIndex: 1000,
          transformOrigin: '0 0',
        }}
      >
        <div style={{ 
          transform: 'translate(-10px, -10px)', // Điều chỉnh vị trí gần con trỏ hơn
        }}>
          <DragOverlayContent 
            activeTask={activeTask}
            getPriorityColor={getPriorityColor}
          />
        </div>
      </DragOverlay>
    </DndContext>
  );
};

export default KanbanBoard;