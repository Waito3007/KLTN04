import React, { useState } from 'react';
import { Card, Row, Col } from 'antd';
import { DndContext, closestCenter } from '@dnd-kit/core';
import { SortableContext, useSortable, arrayMove } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Task } from '../../utils/types';

const SortableTask = ({ task }) => {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: task.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    marginBottom: 8,
  };

  return (
    <Card ref={setNodeRef} style={style} {...attributes} {...listeners}>
      <p>{task.title}</p>
      <p>Người phụ trách: {task.assignee}</p>
    </Card>
  );
};

const TaskBoard = ({ initialTasks = [] }) => {
  const [tasks, setTasks] = useState(initialTasks);

  const onDragEnd = (event) => {
    const { active, over } = event;
    if (active.id !== over.id) {
      setTasks((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  const columns = {
    todo: { title: 'Chờ xử lý', tasks: tasks.filter((task) => task.status === 'todo') },
    inProgress: { title: 'Đang thực hiện', tasks: tasks.filter((task) => task.status === 'inProgress') },
    done: { title: 'Hoàn thành', tasks: tasks.filter((task) => task.status === 'done') },
  };

  return (
    <Card title="Bảng công việc" variant="borderless">
      <DndContext collisionDetection={closestCenter} onDragEnd={onDragEnd}>
        <Row gutter={16}>
          {Object.keys(columns).map((columnId) => (            <Col span={8} key={columnId}>
              <Card title={columns[columnId].title} variant="outlined">
                <SortableContext items={columns[columnId].tasks.map((task) => task.id)}>
                  {columns[columnId].tasks.map((task) => (
                    <SortableTask key={task.id} task={task} />
                  ))}
                </SortableContext>
              </Card>
            </Col>
          ))}
        </Row>
      </DndContext>
    </Card>
  );
};

export default TaskBoard;