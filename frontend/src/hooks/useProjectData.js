// frontend/src/hooks/useProjectData.js
import { useState, useEffect, useCallback } from 'react';
import { message } from 'antd';
import { repositoryAPI, taskAPI, collaboratorAPI, branchAPI } from '../services/api';

// ==================== AUTHENTICATION HELPER ====================
const redirectToLogin = () => {
  window.location.href = '/login';
};

const checkAuthentication = () => {
  const token = localStorage.getItem('access_token');
  if (!token) {
    message.error('🔒 Vui lòng đăng nhập để tiếp tục');
    redirectToLogin();
    return false;
  }
  return true;
};

// ==================== REPOSITORY HOOK ====================
export const useRepositories = (dataSourcePreference = 'auto') => {
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState('database');  const fetchRepositories = useCallback(async () => {
    console.log('fetchRepositories: checking authentication...'); // Debug
    
    // Check authentication before making API calls
    if (!checkAuthentication()) {
      setRepositories([]);
      return;
    }
    
    const token = localStorage.getItem('access_token');
    console.log('fetchRepositories: token preview:', token ? `${token.substring(0, 10)}...` : 'No token'); // Debug

    setLoading(true);
    try {
      let result;
      
      // Handle forced data source preference
      if (dataSourcePreference === 'database') {
        console.log('Fetching from database...'); // Debug
        const data = await repositoryAPI.getFromDatabase();
        console.log('Database result:', data); // Debug
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
      
      console.log('Repositories loaded:', result.data.length, 'repos'); // Debug
      
      // User feedback
      if (result.source === 'github') {
        message.info('📡 Repositories loaded from GitHub API');
      } else {
        message.success('💾 Repositories loaded from local database');
      }    } catch (error) {
      console.error('Error fetching repositories:', error);
      console.error('Error details:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message,
        code: error.code
      });
      setRepositories([]);      if (error.response?.status === 401) {
        message.error('🔒 Phiên đăng nhập đã hết hạn. Đang chuyển hướng đến trang đăng nhập...');
        localStorage.removeItem('access_token');
        setTimeout(() => redirectToLogin(), 1500);
      } else if (error.response?.status === 404 && error.config?.url?.includes('/repositories')) {
        // 404 on repositories endpoint usually means auth issue
        message.error('🔒 Không có quyền truy cập. Vui lòng đăng nhập lại.');
        localStorage.removeItem('access_token');
        setTimeout(() => redirectToLogin(), 1500);
      } else if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
        message.error('❌ Không thể kết nối tới server. Vui lòng kiểm tra server có đang chạy?');
      } else if (error.code === 'ECONNABORTED') {
        message.error('⏱️ Kết nối bị timeout. Vui lòng kiểm tra kết nối mạng và thử lại.');
      } else {
        message.error(error.message || 'Failed to load repositories');
      }
    } finally {
      setLoading(false);
    }}, [dataSourcePreference]);useEffect(() => {
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
export const useProjectData = (options = {}) => {
  const { dataSourcePreference = 'database', preloadedRepositories } = options;
  
  // Individual hooks - chỉ use repositories nếu không có preloaded
  const repositoriesHook = useRepositories(dataSourcePreference);
  
  // Sử dụng preloaded repositories nếu có, fallback đến hook
  const repositories = {
    repositories: preloadedRepositories || repositoriesHook.repositories,
    loading: preloadedRepositories ? false : repositoriesHook.loading,
    dataSource: repositoriesHook.dataSource,
    refetch: repositoriesHook.refetch
  };
  
  const [selectedRepo, setSelectedRepo] = useState(null);
  const tasks = useTasks(selectedRepo, 'database');
  const collaborators = useCollaborators(selectedRepo);
  const [branches, setBranches] = useState([]);
  const [branchesLoading, setBranchesLoading] = useState(false);

  // New function: Auto-sync repository data  // Note: Removed syncRepositoryData function - no longer needed for auto-sync
  // Use individual sync functions (syncBranches, syncCollaborators) instead
  // Load branches from database
  const loadBranches = useCallback(async (repo) => {
    if (!repo) {
      setBranches([]);
      return;
    }

    // Check authentication before loading branches
    if (!checkAuthentication()) {
      setBranches([]);
      return;
    }

    setBranchesLoading(true);
    try {
      console.log(`🌿 Loading branches from database for ${repo.owner.login}/${repo.name}`);
      const branchesData = await branchAPI.getBranches(repo.owner.login, repo.name);
      setBranches(branchesData);
      console.log(`✅ Loaded ${branchesData.length} branches from database`);
    } catch (error) {
      console.error('Failed to load branches:', error);
      setBranches([]);
      
      // Handle authentication errors for branches
      if (error.response?.status === 401 || error.response?.status === 404) {
        message.error('🔒 Không thể tải branches. Vui lòng đăng nhập lại.');
        localStorage.removeItem('access_token');
        setTimeout(() => redirectToLogin(), 1500);
      }
      // Don't show error message for other cases - it's expected to be empty sometimes
    } finally {
      setBranchesLoading(false);
    }
  }, []);
  // Handle repository selection - ONLY load from database, NO auto-sync
  const handleRepoChange = useCallback(async (repoId) => {
    const repo = repositories.repositories.find(r => r.id === repoId);
    setSelectedRepo(repo);
    
    if (repo) {
      console.log(`� Repository selected: ${repo.owner.login}/${repo.name} - Loading from database only`);
      
      // ONLY load branches from database (no auto-sync)
      await loadBranches(repo);
    } else {
      // Clear branches when no repo selected
      setBranches([]);
    }
  }, [repositories.repositories, loadBranches]);
  // Sync branches only
  const syncBranches = useCallback(async () => {
    if (!selectedRepo) return;

    const token = localStorage.getItem('access_token');
    if (!token) {
      message.error('Vui lòng đăng nhập để sync branches');
      return;
    }

    setBranchesLoading(true);
    try {
      console.log(`📂 Syncing branches for ${selectedRepo.owner.login}/${selectedRepo.name}`);
      
      const branchData = await branchAPI.sync(selectedRepo.owner.login, selectedRepo.name);
      setBranches(branchData.branches || []);
      message.success(`✅ Đã sync ${branchData.branches?.length || 0} branches từ GitHub`);
      
    } catch (error) {
      console.error('Failed to sync branches:', error);
      if (error.response?.status === 401) {
        message.error('🔒 Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
      } else {
        message.error('❌ Không thể sync branches từ GitHub');
      }
    } finally {
      setBranchesLoading(false);
    }  }, [selectedRepo]);

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
    deleteTask: tasks.deleteTask,    // Manual refresh functions (no auto-sync)
    refetchRepositories: repositories.refetch,
    refetchTasks: tasks.refetch,
    refetchCollaborators: collaborators.refetch,

    // Manual sync functions (user-initiated only)
    syncBranches,
    syncCollaborators: collaborators.syncCollaborators
  };
};
