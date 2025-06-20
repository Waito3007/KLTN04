import React, { useState } from 'react';
import { 
  Card, 
  Button, 
  Tag, 
  Avatar, 
  Space,
  message,
  Form
} from 'antd';
import { 
  PlusOutlined, 
  UnorderedListOutlined,
  AppstoreOutlined
} from '@ant-design/icons';
import styled from 'styled-components';

// Custom hooks và services
import { useProjectData } from '../../hooks/useProjectData';
import { 
  filterTasks, 
  calculateTaskStats, 
  formatTaskForAPI,
  getStatusIcon,
  getPriorityColor
} from '../../utils/taskUtils.jsx';
import RepoSelector from './ProjectTaskManager/RepoSelector';
import StatisticsPanel from './ProjectTaskManager/StatisticsPanel';
import FiltersPanel from './ProjectTaskManager/FiltersPanel';
import TaskList from './ProjectTaskManager/TaskList';
import TaskModal from './ProjectTaskManager/TaskModal';
import KanbanBoard from './ProjectTaskManager/KanbanBoard';

const TaskCard = styled(Card)`
  margin-bottom: 12px;
  border-radius: 8px;
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
  }
`;

const TaskHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
`;

const TaskActions = styled.div`
  display: flex;
  gap: 8px;
`;

const ProjectTaskManager = () => {
  // ==================== LOCAL STATE (UI ONLY) ====================
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [form] = Form.useForm();
  const [viewMode, setViewMode] = useState(true); // true = Kanban, false = List
  
  // Filter states
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [assigneeFilter, setAssigneeFilter] = useState('all');

  // ==================== CUSTOM HOOK (DATA MANAGEMENT) ====================
  const {
    selectedRepo,
    repositories,
    tasks,
    collaborators,
    repositoriesLoading,
    tasksLoading,
    dataSourceStatus,
    handleRepoChange,
    getAssigneeInfo,
    createTask,
    updateTask,
    updateTaskStatus,
    deleteTask
  } = useProjectData();

  // ==================== COMPUTED VALUES ====================
  const filteredTasks = filterTasks(tasks, {
    searchText,
    statusFilter,
    priorityFilter,
    assigneeFilter
  });

  const taskStats = calculateTaskStats(tasks);

  // ==================== UI HANDLERS ====================
  const showTaskModal = (task = null) => {
    setEditingTask(task);
    setIsModalVisible(true);
    
    if (task) {
      form.setFieldsValue({
        title: task.title,
        description: task.description,
        assignee: task.assignee,
        priority: task.priority,
        dueDate: task.due_date ? new Date(task.due_date) : null
      });
    } else {
      form.resetFields();
    }
  };

  const handleTaskSubmit = async (values) => {
    try {
      const taskData = {
        ...formatTaskForAPI(values),
        status: editingTask ? editingTask.status : 'todo',
        repo_owner: selectedRepo.owner.login,
        repo_name: selectedRepo.name
      };

      if (editingTask) {
        await updateTask(editingTask.id, taskData);
      } else {
        await createTask(taskData);
      }

      setIsModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('Form submission error:', error);
      message.error('Lỗi khi lưu task!');
    }
  };
  // ==================== UI COMPONENTS ====================
  const DataSourceIndicator = () => (
    <div style={{ 
      padding: '8px 12px', 
      background: '#f0f8ff', 
      borderRadius: '6px', 
      marginBottom: '12px',
      fontSize: '12px',
      color: '#666'
    }}>
      <Space size="large">
        <span>
          📊 Repositories: 
          <Tag color={dataSourceStatus.repositories === 'database' ? 'green' : 'orange'} size="small">
            {dataSourceStatus.repositories === 'database' ? '💾 Database' : '📡 GitHub API'}
          </Tag>
        </span>
        {selectedRepo && (
          <>
            <span>
              📝 Tasks: 
              <Tag color={dataSourceStatus.tasks === 'database' ? 'green' : 'blue'} size="small">
                {dataSourceStatus.tasks === 'database' ? '💾 Database' : '📡 Fallback'}
              </Tag>
            </span>
            <span>
              👥 Collaborators: 
              <Tag color={dataSourceStatus.collaborators === 'database' ? 'green' : dataSourceStatus.collaborators === 'github' ? 'orange' : 'purple'} size="small">
                {dataSourceStatus.collaborators === 'database' ? '💾 Database' : 
                 dataSourceStatus.collaborators === 'github' ? '📡 GitHub API' : '🔄 Mixed'}
              </Tag>
            </span>
          </>
        )}
      </Space>
    </div>
  );

  return (
    <Card 
      title="🎯 Quản lý Task Dự án" 
      variant="outlined"
      extra={selectedRepo && (
          <Space>
            <Space.Compact>
              <Button 
                type={viewMode ? "primary" : "default"}
                icon={<AppstoreOutlined />}
                onClick={() => setViewMode(true)}
              >
                Kanban
              </Button>
              <Button 
                type={!viewMode ? "primary" : "default"}
                icon={<UnorderedListOutlined />}
                onClick={() => setViewMode(false)}
              >
                List
              </Button>
            </Space.Compact>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => showTaskModal()}
            >
              Thêm Task
            </Button>
          </Space>
        )
      }
    >
      <DataSourceIndicator />
      
      <div style={{ marginBottom: 16 }}>
        <RepoSelector 
          repositories={repositories}
          selectedRepo={selectedRepo}
          loading={repositoriesLoading}
          handleRepoChange={handleRepoChange}
        />
      </div>
      
      {selectedRepo && (
        <>
          <div style={{ marginBottom: 16, padding: 12, background: '#f5f5f5', borderRadius: 8 }}>
            <Space>
              <Avatar src={selectedRepo.owner.avatar_url} />
              <div>
                <strong>{selectedRepo.owner.login}/{selectedRepo.name}</strong>
                <div style={{ fontSize: 12, color: '#666' }}>
                  {selectedRepo.description || 'Không có mô tả'}
                </div>
              </div>
            </Space>
          </div>
          
          <StatisticsPanel stats={taskStats} />
          
          <Card size="small" style={{ marginBottom: 16 }}>
            <FiltersPanel
              searchText={searchText}
              setSearchText={setSearchText}
              statusFilter={statusFilter}
              setStatusFilter={setStatusFilter}
              priorityFilter={priorityFilter}
              setPriorityFilter={setPriorityFilter}
              assigneeFilter={assigneeFilter}
              setAssigneeFilter={setAssigneeFilter}
              collaborators={collaborators}
              fetchTasks={() => {/* Refresh handled by hook */}}
              tasksLoading={tasksLoading}
              filteredTasks={filteredTasks}
            />
          </Card>
          
          {/* Task View - Kanban or List */}
          {viewMode ? (
            <KanbanBoard
              tasks={filteredTasks}
              getAssigneeInfo={getAssigneeInfo}
              getPriorityColor={getPriorityColor}
              showTaskModal={showTaskModal}
              deleteTask={deleteTask}
              updateTaskStatus={updateTaskStatus}
            />
          ) : (
            <TaskList
              filteredTasks={filteredTasks}
              tasksLoading={tasksLoading}
              getAssigneeInfo={getAssigneeInfo}
              getStatusIcon={getStatusIcon}
              getPriorityColor={getPriorityColor}
              updateTaskStatus={updateTaskStatus}
              showTaskModal={showTaskModal}
              deleteTask={deleteTask}
            />
          )}
        </>
      )}
      
      <TaskModal
        isModalVisible={isModalVisible}
        editingTask={editingTask}
        form={form}
        handleTaskSubmit={handleTaskSubmit}
        setIsModalVisible={setIsModalVisible}
        collaborators={collaborators}
      />
    </Card>
  );
};

export default ProjectTaskManager;