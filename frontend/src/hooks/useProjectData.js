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
    }  }, [dataSourcePreference]);  useEffect(() => {
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
  }, [selectedRepo, fetchTasks]);  useEffect(() => {
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
  const [syncStatus, setSyncStatus] = useState(null);  const fetchCollaborators = useCallback(async () => {
    if (!selectedRepo) {
      console.log('🚫 No selected repo, clearing collaborators');
      setCollaborators([]);
      setSyncStatus(null);
      return;
    }

    const repoKey = `${selectedRepo.owner.login}/${selectedRepo.name}`;
    console.log(`🔄 Fetching collaborators for ${repoKey}`);

    try {
      // 📊 Chỉ lấy từ database (đơn giản)
      const result = await collaboratorAPI.getCollaborators(
        selectedRepo.owner.login, 
        selectedRepo.name
      );
      
      console.log(`📊 Result for ${repoKey}:`, result);
      
      setCollaborators(result.collaborators);
      setDataSource('database');
      setSyncStatus({
        hasSyncedData: result.hasSyncedData,
        message: result.message
      });
      
      console.log(`✅ Loaded ${result.collaborators.length} collaborators for ${repoKey}`);
    } catch (error) {
      console.error(`❌ Error fetching collaborators for ${repoKey}:`, error);
      setCollaborators([]);
      setSyncStatus({
        hasSyncedData: false,
        message: 'Không thể tải collaborators. Vui lòng thử lại.'
      });
    }
  }, [selectedRepo]);  useEffect(() => {
    fetchCollaborators();
  }, [fetchCollaborators]);  // Utility function để get assignee info với fallback đến current user
  const getAssigneeInfo = useCallback((assigneeLogin) => {
    // Tìm trong collaborators trước
    const found = collaborators.find(c => 
      c.login === assigneeLogin || 
      c.github_username === assigneeLogin
    );
    
    if (found) {
      return found;
    }
    
    // Fallback đến current user nếu assignee là chính mình
    try {
      const currentUserProfile = JSON.parse(localStorage.getItem('github_profile') || '{}');
      if (currentUserProfile.login === assigneeLogin) {
        return {
          login: currentUserProfile.login,
          github_username: currentUserProfile.login,
          avatar_url: currentUserProfile.avatar_url,
          display_name: currentUserProfile.name || currentUserProfile.login
        };
      }
    } catch (error) {
      console.warn('Failed to parse github_profile from localStorage:', error);
    }
    
    // Default fallback
    return { 
      login: assigneeLogin, 
      github_username: assigneeLogin,
      avatar_url: null, 
      display_name: assigneeLogin 
    };
  }, [collaborators]);
  // Function to manually clear collaborators data
  const clearCollaborators = useCallback(() => {
    console.log('🧹 Manually clearing collaborators data');
    setCollaborators([]);
    setSyncStatus(null);
    setDataSource('mixed');
  }, []);  // 🔄 Sync collaborators từ GitHub 
  const syncCollaborators = useCallback(async () => {
    if (!selectedRepo) return;

    const repoKey = `${selectedRepo.owner.login}/${selectedRepo.name}`;
    console.log(`🔄 Syncing collaborators for ${repoKey}`);

    try {
      await collaboratorAPI.sync(selectedRepo.owner.login, selectedRepo.name);
      console.log(`✅ Sync completed for ${repoKey}`);
      
      // Refresh data sau khi sync
      await fetchCollaborators();
    } catch (error) {
      console.error(`❌ Sync failed for ${repoKey}:`, error);
      throw error;
    }
  }, [selectedRepo, fetchCollaborators]);

  return {
    collaborators,
    dataSource,
    syncStatus,
    getAssigneeInfo,
    clearCollaborators,
    syncCollaborators,
    refetch: fetchCollaborators
  };
};

// ==================== COMPOSITE HOOK FOR ALL PROJECT DATA ====================
export const useProjectData = (dataSourcePreference = 'database') => {
  // Individual hooks
  const repositories = useRepositories(dataSourcePreference);
  const [selectedRepo, setSelectedRepo] = useState(null);
  const tasks = useTasks(selectedRepo, 'database');
  const collaborators = useCollaborators(selectedRepo);
  const [branches, setBranches] = useState([]);
  const [branchesLoading, setBranchesLoading] = useState(false);

  // New function: Auto-sync repository data
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
      }      // 2. Sync collaborators to database  
      console.log(`👥 Syncing collaborators for ${repo.owner.login}/${repo.name}`);
      
      try {
        const collabResult = await collaboratorAPI.sync(repo.owner.login, repo.name);
        message.success(`✅ Đã sync ${collabResult.saved_collaborators_count || 0} collaborators`);
        
        // Refresh collaborators data
        await collaborators.refetch();
      } catch (collabError) {
        console.error('Failed to sync collaborators:', collabError);
        message.warning('⚠️ Không thể sync collaborators từ GitHub');
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
    }  }, [repositories.repositories, syncRepositoryData]);

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
    
    // Actions
    handleRepoChange,
    getAssigneeInfo: collaborators.getAssigneeInfo,
    
    // Task operations
    createTask: tasks.createTask,
    updateTask: tasks.updateTask,
    updateTaskStatus: tasks.updateTaskStatus,
    deleteTask: tasks.deleteTask,

    // Manual refresh functions (no auto-sync)
    refetchRepositories: repositories.refetch,
    refetchTasks: tasks.refetch,
    refetchCollaborators: collaborators.refetch,
    syncRepositoryData,
    syncCollaborators: collaborators.syncCollaborators
  };
};
