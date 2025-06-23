import React, { useState } from 'react';
import { 
  Card, Button, Space, Form, message, Switch,
  Typography, Divider
} from 'antd';
import { 
  AppstoreOutlined, UnorderedListOutlined, PlusOutlined,
  ReloadOutlined, TeamOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';
import dayjs from 'dayjs';

import { useProjectData } from '../../hooks/useProjectData';
import {
  filterTasks,
  calculateTaskStats,
  formatTaskForAPI
} from '../../utils/taskUtils.jsx';
import RepoSelector from './ProjectTaskManager/RepoSelector';
import StatisticsPanel from './ProjectTaskManager/StatisticsPanel';
import FiltersPanel from './ProjectTaskManager/FiltersPanel';
import TaskList from './ProjectTaskManager/TaskList';
import TaskModal from './ProjectTaskManager/TaskModal';
import KanbanBoard from './ProjectTaskManager/KanbanBoard';
import RepositoryMembers from './RepositoryMembers';

const { Title } = Typography;

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

const ProjectTaskManager = ({ repositories, repoLoading }) => {
  // ==================== LOCAL STATE (UI ONLY) ====================
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [form] = Form.useForm();
  const [viewMode, setViewMode] = useState(true); // true = Kanban, false = List
  const [activeTab, setActiveTab] = useState('tasks'); // 'tasks' or 'members'
  
  // Filter states
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [assigneeFilter, setAssigneeFilter] = useState('all');  // ==================== CUSTOM HOOK (DATA MANAGEMENT) ====================
  const {
    selectedRepo,
    branches,
    tasks,
    collaborators,
    tasksLoading,
    branchesLoading,
    handleRepoChange,
    getAssigneeInfo,
    createTask,
    updateTask,
    updateTaskStatus,    deleteTask,
    syncBranches,
    syncCollaborators
  } = useProjectData({ preloadedRepositories: repositories });

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
        dueDate: task.due_date ? dayjs(task.due_date) : null
      });
    } else {
      form.resetFields();
    }
  };

  const handleTaskSubmit = async (values) => {
    try {
      const taskData = {
        ...formatTaskForAPI(values),
        status: editingTask ? editingTask.status : 'TODO',
        repo_owner: selectedRepo.owner.login,
        repo_name: selectedRepo.name
      };
      if (editingTask) {
        await updateTask(editingTask.id, taskData);
        message.success('✅ Cập nhật task thành công!');
      } else {
        await createTask(taskData);
        message.success('✅ Tạo task thành công!');
      }

      setIsModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('Form submission error:', error);
      message.error('❌ Lỗi khi lưu task!');
    }
  };

  const resetFilters = () => {
    setSearchText('');
    setStatusFilter('all');
    setPriorityFilter('all');
    setAssigneeFilter('all');
  };

  // ==================== RENDER ====================
  return (
    <Card 
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Title level={3} style={{ margin: 0, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            🎯 Quản lý Task Dự án
          </Title>
        </div>
      }
      style={{ minHeight: '80vh' }}      extra={selectedRepo && (
        <Space>
          {/* Tab Switcher */}
          <Space.Compact>
            <Button 
              type={activeTab === 'tasks' ? "primary" : "default"}
              onClick={() => setActiveTab('tasks')}
              style={{ borderRadius: '6px 0 0 6px' }}
            >
              📋 Tasks
            </Button>
            <Button 
              type={activeTab === 'members' ? "primary" : "default"}
              icon={<TeamOutlined />}
              onClick={() => setActiveTab('members')}
              style={{ borderRadius: '0 6px 6px 0' }}
            >
              👥 Members
            </Button>
          </Space.Compact>

          {/* Task View Mode (only show when on tasks tab) */}
          {activeTab === 'tasks' && (
            <Space.Compact>
              <Button 
                type={viewMode ? "primary" : "default"}
                icon={<AppstoreOutlined />}
                onClick={() => setViewMode(true)}
                style={{ borderRadius: '6px 0 0 6px' }}
              >
                Kanban
              </Button>
              <Button 
                type={!viewMode ? "primary" : "default"}
                icon={<UnorderedListOutlined />}
                onClick={() => setViewMode(false)}
                style={{ borderRadius: '0 6px 6px 0' }}
              >
                List
              </Button>
            </Space.Compact>
          )}
          
          {activeTab === 'tasks' && (
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => showTaskModal()}
              disabled={!selectedRepo}
              style={{ 
                borderRadius: 6,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none'
              }}
            >
              Tạo Task
            </Button>
          )}
        </Space>
      )}
    >
      {/* Repository Selector - Clean & Simple */}      <RepoSelector 
        repositories={repositories}
        selectedRepo={selectedRepo}
        loading={repoLoading}
        handleRepoChange={handleRepoChange}
        branches={branches}
        collaborators={collaborators}
        branchesLoading={branchesLoading}
        onSyncBranches={syncBranches}
        onSyncCollaborators={syncCollaborators}
      />{/* Tab Content - Conditional Rendering */}
      {selectedRepo && (
        <>
          {activeTab === 'tasks' && (
            <>
              {/* Statistics Panel */}
              <StatisticsPanel 
                stats={taskStats}
                selectedRepo={selectedRepo}
                collaborators={collaborators}
              />
              <Divider />
              {/* Filters Panel */}
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
                filteredTasks={filteredTasks}
                tasksLoading={tasksLoading}
                resetFilters={resetFilters}
              />
              <Divider />
              {/* Tasks Display */}
              {viewMode ? (
                <KanbanBoard 
                  tasks={filteredTasks}
                  getAssigneeInfo={getAssigneeInfo}
                  getPriorityColor={(priority) => {
                    switch (priority) {
                      case 'urgent': return '#ff4d4f';
                      case 'high': return '#ff7a45';
                      case 'medium': return '#faad14';
                      case 'low': return '#52c41a';
                      default: return '#1890ff';
                    }
                  }}
                  showTaskModal={showTaskModal}
                  deleteTask={deleteTask}
                  updateTaskStatus={updateTaskStatus}
                />
              ) : (
                <TaskList 
                  filteredTasks={filteredTasks}
                  tasksLoading={tasksLoading}
                  getAssigneeInfo={getAssigneeInfo}
                  getStatusIcon={(status) => {
                    switch (status) {
                      case 'todo': return '📋';
                      case 'in_progress': return '⚡';
                      case 'done': return '✅';
                      default: return '📋';
                    }
                  }}
                  getPriorityColor={(priority) => {
                    switch (priority) {
                      case 'urgent': return '#ff4d4f';
                      case 'high': return '#ff7a45';
                      case 'medium': return '#faad14';
                      case 'low': return '#52c41a';
                      default: return '#1890ff';
                    }
                  }}
                  updateTaskStatus={updateTaskStatus}
                  showTaskModal={showTaskModal}
                  deleteTask={deleteTask}
                />
              )}
            </>
          )}
          {activeTab === 'members' && (
            <RepositoryMembers selectedRepo={selectedRepo} />
          )}
        </>
      )}
      {/* Task Modal */}
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