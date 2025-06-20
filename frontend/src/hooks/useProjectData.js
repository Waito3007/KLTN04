// frontend/src/hooks/useProjectData.js
import { useState, useEffect, useCallback } from 'react';
import { message } from 'antd';
import { repositoryAPI, taskAPI, collaboratorAPI } from '../services/api';

// ==================== REPOSITORY HOOK ====================
export const useRepositories = (dataSourcePreference = 'auto') => {
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
      let result;
      
      // Handle forced data source preference
      if (dataSourcePreference === 'database') {
        const data = await repositoryAPI.getFromDatabase();
        result = { data, source: 'database' };
      } else if (dataSourcePreference === 'github') {
        const data = await repositoryAPI.getFromGitHub();
        result = { data, source: 'github' };
      } else {
        // Auto mode - intelligent fallback
        result = await repositoryAPI.getIntelligent();
      }
      
      setRepositories(result.data);
      setDataSource(result.source);
      
      // User feedback
      if (result.source === 'github') {
        message.info('📡 Repositories loaded from GitHub API');
      } else {
        message.success('💾 Repositories loaded from local database');
      }
    } catch (error) {
      console.error('Error fetching repositories:', error);
      setRepositories([]);
      message.error(error.message);
    } finally {
      setLoading(false);
    }  }, [dataSourcePreference]);

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
export const useTasks = (selectedRepo, dataSourcePreference = 'auto') => {
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
      let result;
      
      // Handle forced data source preference
      if (dataSourcePreference === 'database') {
        const data = await taskAPI.getByRepo(selectedRepo.owner.login, selectedRepo.name);
        result = { data, source: 'database' };
      } else if (dataSourcePreference === 'fallback') {
        const data = await taskAPI.getAll(selectedRepo.owner.login, selectedRepo.name);
        result = { data, source: 'fallback' };
      } else {
        // Auto mode - intelligent fallback
        result = await taskAPI.getIntelligent(
          selectedRepo.owner.login,
          selectedRepo.name
        );
      }
      
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
  }, [selectedRepo, dataSourcePreference]);

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
export const useCollaborators = (selectedRepo, dataSourcePreference = 'auto') => {
  const [collaborators, setCollaborators] = useState([]);
  const [dataSource, setDataSource] = useState('mixed');

  const fetchCollaborators = useCallback(async () => {
    if (!selectedRepo) {
      setCollaborators([]);
      return;
    }

    console.log(`🔄 fetchCollaborators called for ${selectedRepo.owner.login}/${selectedRepo.name}`, {
      dataSourcePreference,
      selectedRepo: selectedRepo.owner.login + '/' + selectedRepo.name
    });

    try {
      let result;
      
      // Handle forced data source preference
      if (dataSourcePreference === 'database') {
        console.log('📊 Using database preference');
        const data = await collaboratorAPI.getFromBackend(selectedRepo.owner.login, selectedRepo.name);
        console.log('📊 Database result:', data);
        result = { data, source: 'database' };
      } else if (dataSourcePreference === 'github') {
        console.log('📡 Using GitHub preference');
        const contributors = await collaboratorAPI.getFromGitHub(selectedRepo.owner.login, selectedRepo.name);
        const ownerEntry = {
          login: selectedRepo.owner.login,
          avatar_url: selectedRepo.owner.avatar_url,
          type: 'Owner',
          contributions: 0
        };
        const uniqueCollaborators = [
          ownerEntry,
          ...contributors.filter(c => c.login !== selectedRepo.owner.login)
        ];
        result = { data: uniqueCollaborators, source: 'github' };
      } else {
        console.log('🔄 Using auto/intelligent mode');
        // Auto mode - intelligent fallback
        result = await collaboratorAPI.getIntelligent(
          selectedRepo.owner.login,
          selectedRepo.name,
          selectedRepo.owner
        );
        console.log('🔄 Intelligent result:', result);
      }
      
      console.log(`🎯 Setting collaborators:`, result.data);
      setCollaborators(result.data);
      setDataSource(result.source);
      
      console.log(`✅ Loaded ${result.data.length} collaborators from ${result.source}`);
    } catch (error) {
      console.error('❌ Error fetching collaborators:', error);
      setCollaborators([]);
    }
  }, [selectedRepo, dataSourcePreference]);

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
export const useProjectData = (preferences = {}) => {
  const {
    repoDataSource = 'auto',
    taskDataSource = 'auto', 
    collaboratorDataSource = 'auto'
  } = preferences;
  
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [branches, setBranches] = useState([]);
  const [branchesLoading, setBranchesLoading] = useState(false);
  
  const repositories = useRepositories(repoDataSource);
  const tasks = useTasks(selectedRepo, taskDataSource);
  const collaborators = useCollaborators(selectedRepo, collaboratorDataSource);

  // Computed data source status
  const dataSourceStatus = {
    repositories: repositories.dataSource,
    tasks: tasks.dataSource,
    collaborators: collaborators.dataSource
  };  // New function: Auto-sync repository data
  const syncRepositoryData = useCallback(async (repo) => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      setBranchesLoading(true);
      
      // 1. Sync branches to database
      console.log(`📂 Syncing branches for ${repo.owner.login}/${repo.name}`);
      const branchResponse = await fetch(`http://localhost:8000/api/github/${repo.owner.login}/${repo.name}/sync-branches`, {
        method: 'POST',
        headers: {
          'Authorization': `token ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (branchResponse.ok) {
        const branchData = await branchResponse.json();
        setBranches(branchData.branches || []);
        message.success(`✅ Đã sync ${branchData.branches?.length || 0} branches`);
      }

      // 2. Sync collaborators to database  
      console.log(`👥 Syncing collaborators for ${repo.owner.login}/${repo.name}`);
      const collabResponse = await fetch(`http://localhost:8000/api/github/${repo.owner.login}/${repo.name}/sync-collaborators`, {
        method: 'POST', 
        headers: {
          'Authorization': `token ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (collabResponse.ok) {
        const collabData = await collabResponse.json();
        message.success(`✅ Đã sync ${collabData.saved_collaborators_count || 0} collaborators`);
        
        // Refresh collaborators data
        await collaborators.refetch();
      }

    } catch (error) {
      console.error('Sync error:', error);
      message.warning('⚠️ Một số dữ liệu không sync được, sẽ dùng cache');
    } finally {
      setBranchesLoading(false);
    }
  }, [collaborators]);

  // Enhanced handleRepoChange with auto-sync
  const handleRepoChange = useCallback(async (repoId) => {
    const repo = repositories.repositories.find(r => r.id === repoId);
    setSelectedRepo(repo);
    
    if (repo) {
      console.log(`🔄 Repository selected: ${repo.owner.login}/${repo.name}`);
      
      // Trigger background sync for branches & collaborators
      await syncRepositoryData(repo);
    }
  }, [repositories.repositories, syncRepositoryData]);

  return {
    // States
    selectedRepo,
    branches,
    
    // Data
    repositories: repositories.repositories,
    tasks: tasks.tasks,
    collaborators: collaborators.collaborators,
    
    // Loading states
    repositoriesLoading: repositories.loading,
    tasksLoading: tasks.loading,
    branchesLoading,
    
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
    refetchCollaborators: collaborators.refetch,
    syncRepositoryData
  };
};
