import React from 'react';
import { List, Empty, Spin } from 'antd';
import TaskCard from './TaskCard';

const TaskList = ({ filteredTasks = [], tasksLoading = false, getAssigneeInfo, getStatusIcon, getPriorityColor, updateTaskStatus, showTaskModal, deleteTask }) => {
  // Double safety check
  const safeTasks = Array.isArray(filteredTasks) ? filteredTasks : [];
  
  return (
    <Spin spinning={tasksLoading}>
      {safeTasks.length === 0 ? (
        <Empty 
          description="Chưa có task nào cho repository này"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <List
          dataSource={safeTasks}
        renderItem={task => (
          <TaskCard
            task={task}
            getAssigneeInfo={getAssigneeInfo}
            getStatusIcon={getStatusIcon}
            getPriorityColor={getPriorityColor}            updateTaskStatus={updateTaskStatus}
            showTaskModal={showTaskModal}
            deleteTask={deleteTask}
          />
        )}
      />
    )}
  </Spin>
  );
};

export default TaskList;
