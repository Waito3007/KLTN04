import React, { useState, useEffect, useCallback } from 'react';
import { 
  Card, 
  Select, 
  List, 
  Button, 
  Modal, 
  Input, 
  DatePicker, 
  Tag, 
  Avatar, 
  Space,
  Tooltip,
  message,
  Empty,
  Spin,
  Row,
  Col,
  Statistic,
  Progress,
  Badge,
  Dropdown,
  Menu,
  Form
} from 'antd';
import { 
  PlusOutlined, 
  UserOutlined, 
  CalendarOutlined, 
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  SearchOutlined,
  FilterOutlined,
  BarChartOutlined,
  ReloadOutlined,
  UnorderedListOutlined,
  AppstoreOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import axios from 'axios';
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
  const [repositories, setRepositories] = useState([]);
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [filteredTasks, setFilteredTasks] = useState([]);
  const [collaborators, setCollaborators] = useState([]);
  const [loading, setLoading] = useState(false);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [form] = Form.useForm();
    // Filter states
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [assigneeFilter, setAssigneeFilter] = useState('all');

  // View state - true for Kanban, false for List
  const [viewMode, setViewMode] = useState(true); // Default to Kanban

  const fetchRepositories = useCallback(async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/api/github/repos', {
        headers: { Authorization: `token ${token}` },
      });
      setRepositories(response.data || []);
    } catch (error) {
      console.error('Lỗi khi tải repositories:', error);
      message.error('Không thể tải danh sách repository!');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchTasks = useCallback(async () => {
    if (!selectedRepo) return;

    try {
      setTasksLoading(true);
      const response = await axios.get(
        `http://localhost:8000/api/projects/${selectedRepo.owner.login}/${selectedRepo.name}/tasks`,
        {
          headers: { Authorization: `token ${localStorage.getItem('access_token')}` },
        }
      );
      setTasks(response.data || []);
      console.log('Tasks fetched from API:', response.data);
    } catch {
      console.log('API chưa có dữ liệu, sử dụng dữ liệu mẫu cho tasks');
      // Dữ liệu mẫu khi API chưa có - sử dụng owner thật của repo
      const ownerLogin = selectedRepo.owner.login;
      setTasks([
        {
          id: 1,
          title: 'Fix authentication bug',
          description: 'Sửa lỗi đăng nhập không thành công trong hệ thống',
          assignee: ownerLogin,
          status: 'todo',
          priority: 'high',
          due_date: '2025-06-20',
          created_at: '2025-06-10'
        },
        {
          id: 2,
          title: 'Update documentation',
          description: 'Cập nhật tài liệu hướng dẫn API và README',
          assignee: ownerLogin,
          status: 'in_progress',
          priority: 'medium',
          due_date: '2025-06-25',
          created_at: '2025-06-11'
        },
        {
          id: 3,
          title: 'Optimize database queries',
          description: 'Tối ưu hóa các truy vấn database để cải thiện performance',
          assignee: ownerLogin,
          status: 'todo',
          priority: 'low',
          due_date: '2025-06-30',
          created_at: '2025-06-12'
        }
      ]);
    } finally {
      setTasksLoading(false);
    }
  }, [selectedRepo]);
  const fetchCollaborators = useCallback(async () => {
    if (!selectedRepo) return;

    try {
      // Thử lấy contributors từ GitHub API (ít bị hạn chế hơn collaborators)
      const token = localStorage.getItem('access_token');
      const response = await axios.get(
        `https://api.github.com/repos/${selectedRepo.owner.login}/${selectedRepo.name}/contributors`,
        {
          headers: { 
            Authorization: `token ${token}`,
            Accept: 'application/vnd.github.v3+json'
          },
        }
      );
      
      // Thêm owner vào đầu danh sách
      const ownerData = {
        login: selectedRepo.owner.login,
        avatar_url: selectedRepo.owner.avatar_url,
        type: 'Owner',
        contributions: 0
      };
      
      // Lọc và format contributors
      const contributors = response.data.slice(0, 10).map(contributor => ({
        login: contributor.login,
        avatar_url: contributor.avatar_url,
        type: contributor.login === selectedRepo.owner.login ? 'Owner' : 'Contributor',
        contributions: contributor.contributions
      }));
      
      // Đảm bảo owner luôn ở đầu danh sách
      const uniqueCollaborators = [
        ownerData,
        ...contributors.filter(c => c.login !== selectedRepo.owner.login)
      ];
      
      setCollaborators(uniqueCollaborators);
      console.log(`✅ Loaded ${uniqueCollaborators.length} contributors for ${selectedRepo.name}`);
      
    } catch (error) {
      console.log('Không thể lấy contributors từ GitHub API, sử dụng fallback:', error.message);
      
      // Fallback: hiển thị owner và một số thành viên từ backend nếu có
      try {
        const backupResponse = await axios.get(
          `http://localhost:8000/api/github/${selectedRepo.owner.login}/${selectedRepo.name}/collaborators`,
          {
            headers: { Authorization: `token ${localStorage.getItem('access_token')}` },
          }
        );
        setCollaborators(backupResponse.data || []);
      } catch {
        // Last fallback: chỉ hiển thị owner
        setCollaborators([
          {
            login: selectedRepo.owner.login,
            avatar_url: selectedRepo.owner.avatar_url,
            type: 'Owner',
            contributions: 0
          }
        ]);
      }
    }
  }, [selectedRepo]);

  // Load repositories khi component mount
  useEffect(() => {
    fetchRepositories();
  }, [fetchRepositories]);

  // Load tasks và collaborators khi chọn repo
  useEffect(() => {
    if (selectedRepo) {
      fetchTasks();
      fetchCollaborators();
    }
  }, [selectedRepo, fetchTasks, fetchCollaborators]);

  const handleRepoChange = (repoId) => {
    const repo = repositories.find(r => r.id === repoId);
    setSelectedRepo(repo);
    setTasks([]);
  };

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
        ...values,
        due_date: values.dueDate ? values.dueDate.format('YYYY-MM-DD') : null,
        status: editingTask ? editingTask.status : 'todo',
        repo_owner: selectedRepo.owner.login,
        repo_name: selectedRepo.name
      };

      if (editingTask) {
        // Update task via API
        try {
          await axios.put(
            `http://localhost:8000/api/projects/${selectedRepo.owner.login}/${selectedRepo.name}/tasks/${editingTask.id}`,
            taskData,
            {
              headers: { Authorization: `token ${localStorage.getItem('access_token')}` },
            }
          );
          // Refresh tasks from server
          await fetchTasks();
          message.success('Cập nhật task thành công!');
        } catch (apiError) {
          console.log('API call failed, using local update:', apiError);
          // Fallback to local update
          const updatedTasks = tasks.map(task => 
            task.id === editingTask.id ? { ...task, ...taskData } : task
          );
          setTasks(updatedTasks);
          message.success('Cập nhật task thành công (local)!');
        }
      } else {
        // Create new task via API
        try {
          await axios.post(
            `http://localhost:8000/api/projects/${selectedRepo.owner.login}/${selectedRepo.name}/tasks`,
            taskData,
            {
              headers: { Authorization: `token ${localStorage.getItem('access_token')}` },
            }
          );
          // Refresh tasks from server
          await fetchTasks();
          message.success('Tạo task mới thành công!');
        } catch (apiError) {
          console.log('API call failed, using local creation:', apiError);
          // Fallback to local creation
          const newTask = {
            id: Date.now(),
            ...taskData,
            created_at: new Date().toISOString().split('T')[0]
          };
          setTasks([...tasks, newTask]);
          message.success('Tạo task mới thành công (local)!');
        }
      }

      setIsModalVisible(false);
      form.resetFields();
    } catch (formError) {
      console.error('Form submission error:', formError);
      message.error('Lỗi khi lưu task!');
    }
  };  const updateTaskStatus = async (taskId, newStatus) => {
    try {
      console.log(`Updating task ${taskId} to status ${newStatus}`);
      const taskToUpdate = tasks.find(t => t.id === taskId);
      if (!taskToUpdate) {
        console.error(`Task with ID ${taskId} not found`);
        return;
      }

      const updatedTaskData = { ...taskToUpdate, status: newStatus };
      
      try {
        await axios.put(
          `http://localhost:8000/api/projects/${selectedRepo.owner.login}/${selectedRepo.name}/tasks/${taskId}`,
          updatedTaskData,
          {
            headers: { Authorization: `token ${localStorage.getItem('access_token')}` },
          }
        );
        console.log('Task updated successfully via API');
        // Refresh tasks from server
        await fetchTasks();
        message.success('Cập nhật trạng thái thành công!');
      } catch (apiError) {
        console.log('API call failed, using local update:', apiError);
        // Fallback to local update
        const updatedTasks = tasks.map(task => 
          task.id === taskId ? { ...task, status: newStatus } : task
        );
        console.log('Updated tasks locally:', updatedTasks);
        setTasks(updatedTasks);
        message.success('Cập nhật trạng thái thành công (local)!');
      }
    } catch (error) {
      console.error('Error updating task status:', error);
      message.error('Lỗi khi cập nhật trạng thái!');
    }
  };

  const deleteTask = async (taskId) => {
    try {
      try {
        await axios.delete(
          `http://localhost:8000/api/projects/${selectedRepo.owner.login}/${selectedRepo.name}/tasks/${taskId}`,
          {
            headers: { Authorization: `token ${localStorage.getItem('access_token')}` },
          }
        );
        // Refresh tasks from server
        await fetchTasks();
        message.success('Xóa task thành công!');
      } catch (apiError) {
        console.log('API call failed, using local delete:', apiError);
        // Fallback to local delete
        const updatedTasks = tasks.filter(task => task.id !== taskId);
        setTasks(updatedTasks);
        message.success('Xóa task thành công (local)!');
      }
    } catch (error) {
      console.error('Error deleting task:', error);
      message.error('Lỗi khi xóa task!');
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'todo': return <ClockCircleOutlined style={{ color: '#faad14' }} />;
      case 'in_progress': return <ExclamationCircleOutlined style={{ color: '#1890ff' }} />;
      case 'done': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      default: return <ClockCircleOutlined />;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return '#f5222d';
      case 'medium': return '#fa8c16';
      case 'low': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  const getAssigneeInfo = (assigneeLogin) => {
    return collaborators.find(c => c.login === assigneeLogin) || 
           { login: assigneeLogin, avatar_url: null };
  };

  // Filter and search functions
  const applyFilters = useCallback(() => {
    let filtered = [...tasks];
    
    // Search filter
    if (searchText) {
      filtered = filtered.filter(task => 
        task.title.toLowerCase().includes(searchText.toLowerCase()) ||
        task.description?.toLowerCase().includes(searchText.toLowerCase()) ||
        task.assignee.toLowerCase().includes(searchText.toLowerCase())
      );
    }
    
    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(task => task.status === statusFilter);
    }
    
    // Priority filter
    if (priorityFilter !== 'all') {
      filtered = filtered.filter(task => task.priority === priorityFilter);
    }
    
    // Assignee filter
    if (assigneeFilter !== 'all') {
      filtered = filtered.filter(task => task.assignee === assigneeFilter);
    }
    
    setFilteredTasks(filtered);
  }, [tasks, searchText, statusFilter, priorityFilter, assigneeFilter]);

  // Statistics calculation
  const getTaskStats = useCallback(() => {
    const total = tasks.length;
    const completed = tasks.filter(t => t.status === 'done').length;
    const inProgress = tasks.filter(t => t.status === 'in_progress').length;
    const todo = tasks.filter(t => t.status === 'todo').length;
    const highPriority = tasks.filter(t => t.priority === 'high').length;
    
    return {
      total,
      completed,
      inProgress,
      todo,
      highPriority,
      completionRate: total > 0 ? Math.round((completed / total) * 100) : 0
    };
  }, [tasks]);

  // Apply filters whenever dependencies change
  useEffect(() => {
    applyFilters();
  }, [applyFilters]);  return (
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
      <div style={{ marginBottom: 16 }}>
        <RepoSelector 
          repositories={repositories}
          selectedRepo={selectedRepo}
          loading={loading}
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
          <StatisticsPanel stats={getTaskStats()} />
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
              fetchTasks={fetchTasks}
              tasksLoading={tasksLoading}
              filteredTasks={filteredTasks}
            />          </Card>
          
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