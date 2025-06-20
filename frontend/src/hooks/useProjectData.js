// frontend/src/hooks/useProjectData.js
import { useState, useEffect, useCallback } from 'react';
import { message } from 'antd';
import { repositoryAPI, taskAPI, collaboratorAPI } from '../services/api';

// ==================== REPOSITORY HOOK ====================
export const useRepositories = () => {
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState('database');

  const fetchRepositories = useCallback(async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      message.error('Vui lòng đăng nhập để tiếp tục');
      return;
    }

    setLoading(true);
    try {
      const result = await repositoryAPI.getIntelligent();
      setRepositories(result.data);
      setDataSource(result.source);
      
      // User feedback
      if (result.source === 'github') {
        message.info('📡 Repositories loaded from GitHub API (database unavailable)');
      } else {
        message.success('💾 Repositories loaded from local database');
      }
    } catch (error) {
      console.error('Error fetching repositories:', error);
      setRepositories([]);
      message.error(error.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRepositories();
  }, [fetchRepositories]);

  return {
    repositories,
    loading,
    dataSource,
    refetch: fetchRepositories
  };
};

// ==================== TASKS HOOK ====================
export const useTasks = (selectedRepo) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState('database');

  const fetchTasks = useCallback(async () => {
    if (!selectedRepo) {
      setTasks([]);
      return;
    }

    setLoading(true);
    try {
      const result = await taskAPI.getIntelligent(
        selectedRepo.owner.login,
        selectedRepo.name
      );
      
      setTasks(result.data);
      setDataSource(result.source);
      
      if (result.data.length === 0) {
        message.info('Không có tasks nào cho repository này');
      } else if (result.source === 'fallback') {
        message.info('📡 Tasks loaded from general database');
      }
    } catch (error) {
      console.error('Error fetching tasks:', error);
      setTasks([]);
      message.error('Cannot load tasks');
    } finally {
      setLoading(false);
    }
  }, [selectedRepo]);

  // API operations với fallback local
  const createTask = useCallback(async (taskData) => {
    try {
      await taskAPI.create(
        selectedRepo.owner.login,
        selectedRepo.name,
        taskData
      );
      await fetchTasks(); // Refresh từ server
      message.success('Tạo task mới thành công!');
    } catch (error) {
      console.log('API failed, using local creation:', error);
      // Local fallback
      const newTask = {
        id: Date.now(),
        ...taskData,
        created_at: new Date().toISOString().split('T')[0]
      };
      setTasks(prev => [...prev, newTask]);
      message.success('Tạo task mới thành công (local)!');
    }
  }, [selectedRepo, fetchTasks]);

  const updateTask = useCallback(async (taskId, taskData) => {
    try {
      await taskAPI.update(
        selectedRepo.owner.login,
        selectedRepo.name,
        taskId,
        taskData
      );
      await fetchTasks(); // Refresh từ server
      message.success('Cập nhật task thành công!');
    } catch (error) {
      console.log('API failed, using local update:', error);
      // Local fallback
      setTasks(prev => prev.map(task => 
        task.id === taskId ? { ...task, ...taskData } : task
      ));
      message.success('Cập nhật task thành công (local)!');
    }
  }, [selectedRepo, fetchTasks]);

  const updateTaskStatus = useCallback(async (taskId, newStatus) => {
    const taskToUpdate = tasks.find(t => t.id === taskId);
    if (!taskToUpdate) return;

    await updateTask(taskId, { ...taskToUpdate, status: newStatus });
  }, [tasks, updateTask]);

  const deleteTask = useCallback(async (taskId) => {
    try {
      await taskAPI.delete(
        selectedRepo.owner.login,
        selectedRepo.name,
        taskId
      );
      await fetchTasks(); // Refresh từ server
      message.success('Xóa task thành công!');
    } catch (error) {
      console.log('API failed, using local delete:', error);
      // Local fallback
      setTasks(prev => prev.filter(task => task.id !== taskId));
      message.success('Xóa task thành công (local)!');
    }
  }, [selectedRepo, fetchTasks]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  return {
    tasks,
    loading,
    dataSource,
    createTask,
    updateTask,
    updateTaskStatus,
    deleteTask,
    refetch: fetchTasks
  };
};

// ==================== COLLABORATORS HOOK ====================
export const useCollaborators = (selectedRepo) => {
  const [collaborators, setCollaborators] = useState([]);
  const [dataSource, setDataSource] = useState('mixed');

  const fetchCollaborators = useCallback(async () => {
    if (!selectedRepo) {
      setCollaborators([]);
      return;
    }

    try {
      const result = await collaboratorAPI.getIntelligent(
        selectedRepo.owner.login,
        selectedRepo.name,
        selectedRepo.owner
      );
      
      setCollaborators(result.data);
      setDataSource(result.source);
      
      console.log(`✅ Loaded ${result.data.length} collaborators from ${result.source}`);
    } catch (error) {
      console.error('Error fetching collaborators:', error);
      setCollaborators([]);
    }
  }, [selectedRepo]);

  useEffect(() => {
    fetchCollaborators();
  }, [fetchCollaborators]);

  // Utility function để get assignee info
  const getAssigneeInfo = useCallback((assigneeLogin) => {
    return collaborators.find(c => c.login === assigneeLogin) || 
           { login: assigneeLogin, avatar_url: null };
  }, [collaborators]);

  return {
    collaborators,
    dataSource,
    getAssigneeInfo,
    refetch: fetchCollaborators
  };
};

// ==================== COMPOSITE HOOK FOR ALL PROJECT DATA ====================
export const useProjectData = () => {
  const [selectedRepo, setSelectedRepo] = useState(null);
  
  const repositories = useRepositories();
  const tasks = useTasks(selectedRepo);
  const collaborators = useCollaborators(selectedRepo);

  // Computed data source status
  const dataSourceStatus = {
    repositories: repositories.dataSource,
    tasks: tasks.dataSource,
    collaborators: collaborators.dataSource
  };

  const handleRepoChange = useCallback((repoId) => {
    const repo = repositories.repositories.find(r => r.id === repoId);
    setSelectedRepo(repo);
  }, [repositories.repositories]);

  return {
    // States
    selectedRepo,
    
    // Data
    repositories: repositories.repositories,
    tasks: tasks.tasks,
    collaborators: collaborators.collaborators,
    
    // Loading states
    repositoriesLoading: repositories.loading,
    tasksLoading: tasks.loading,
    
    // Data sources
    dataSourceStatus,
    
    // Actions
    handleRepoChange,
    getAssigneeInfo: collaborators.getAssigneeInfo,
    
    // Task operations
    createTask: tasks.createTask,
    updateTask: tasks.updateTask,
    updateTaskStatus: tasks.updateTaskStatus,
    deleteTask: tasks.deleteTask,
    
    // Refresh functions
    refetchRepositories: repositories.refetch,
    refetchTasks: tasks.refetch,
    refetchCollaborators: collaborators.refetch
  };
};
