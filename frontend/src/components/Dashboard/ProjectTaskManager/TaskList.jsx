import React from 'react';
import { List, Empty, Spin } from 'antd';
import TaskCard from './TaskCard';

const TaskList = ({ filteredTasks, tasksLoading, getAssigneeInfo, getStatusIcon, getPriorityColor, updateTaskStatus, showTaskModal, deleteTask }) => (
  <Spin spinning={tasksLoading}>
    {filteredTasks.length === 0 ? (
      <Empty 
        description="Chưa có task nào cho repository này"
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    ) : (
      <List
        dataSource={filteredTasks}
        renderItem={task => (
          <TaskCard
            task={task}
            getAssigneeInfo={getAssigneeInfo}
            getStatusIcon={getStatusIcon}
            getPriorityColor={getPriorityColor}
            updateTaskStatus={updateTaskStatus}
            showTaskModal={showTaskModal}
            deleteTask={deleteTask}
          />
        )}
      />
    )}
  </Spin>
);

export default TaskList;
