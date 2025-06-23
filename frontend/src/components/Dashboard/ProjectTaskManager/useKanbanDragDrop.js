// useKanbanDragDrop.js
import { useState } from 'react';
import { useSensor, useSensors, PointerSensor, KeyboardSensor } from '@dnd-kit/core';
import { DRAG_CONFIG } from './kanbanConstants';

export const useKanbanDragDrop = ({ tasks, updateTaskStatus, columns }) => {
  const [activeId, setActiveId] = useState(null);
  // Cấu hình sensors để cursor gần hơn với task
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: DRAG_CONFIG.ACTIVATION_DISTANCE,
      },
      // Điều chỉnh để cursor gần task hơn
      coordinateGetter: (event) => ({
        x: event.clientX,
        y: event.clientY,
      }),
    }),
    useSensor(KeyboardSensor)
  );

  const handleDragStart = (event) => {
    setActiveId(event.active.id);
  };
  const handleDragEnd = (event) => {
    const { active, over } = event;
    setActiveId(null);
    
    if (!over) return;

    const draggedTaskId = active.id;
    const targetId = over.id;

    try {
      // Tìm task đang được kéo
      const draggedTask = tasks.find(t => t.id === draggedTaskId);
      if (!draggedTask) {
        console.error(`Dragged task with ID ${draggedTaskId} not found`);
        return;
      }

      // Kiểm tra xem target có phải là column không
      const targetColumn = columns.find(col => col.id === targetId);
      
      if (targetColumn) {
        // Kéo vào column
        if (draggedTask.status !== targetColumn.id) {
          console.log(`Moving task ${draggedTaskId} to column ${targetColumn.id}`);
          updateTaskStatus(draggedTaskId, targetColumn.id);
        }
      } else {
        // Kéo vào task khác - tìm column chứa task đó
        const targetTask = tasks.find(t => t.id === targetId);
        if (targetTask && draggedTask.status !== targetTask.status) {
          console.log(`Moving task ${draggedTaskId} to column ${targetTask.status} (via task)`);
          updateTaskStatus(draggedTaskId, targetTask.status);
        }
      }
    } catch (error) {
      console.error('Error in handleDragEnd:', error);
      // Không show message error để tránh làm crash UI
    }
  };

  const activeTask = activeId ? tasks.find(t => t.id === activeId) : null;

  return {
    sensors,
    activeId,
    activeTask,
    handleDragStart,
    handleDragEnd,
    dropAnimation: DRAG_CONFIG.DROP_ANIMATION
  };
};
