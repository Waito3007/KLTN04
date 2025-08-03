// frontend/src/hooks/useSync.js
import { useState, useCallback } from 'react';
import { message } from 'antd';
import { syncAPI } from '../services/api';

export const useSync = () => {
  const [loading, setLoading] = useState(false);
  const [syncResults, setSyncResults] = useState(null);
  const [error, setError] = useState(null);

  const clearResults = useCallback(() => {
    setSyncResults(null);
    setError(null);
  }, []);

  const syncAll = useCallback(async (owner, repoName) => {
    setLoading(true);
    setError(null);
    setSyncResults(null);

    try {
      message.loading('Đang đồng bộ toàn bộ repository...', 0);
      const result = await syncAPI.syncAll(owner, repoName);
      
      setSyncResults(result);
      message.destroy();
      message.success('Đồng bộ toàn bộ hoàn tất!');
      
      return result;
    } catch (err) {
      console.error('Sync all error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Có lỗi xảy ra khi đồng bộ';
      setError(errorMessage);
      message.destroy();
      message.error(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const syncBasic = useCallback(async (owner, repoName) => {
    setLoading(true);
    setError(null);
    setSyncResults(null);

    try {
      message.loading('Đang đồng bộ cơ bản...', 0);
      const result = await syncAPI.syncBasic(owner, repoName);
      
      setSyncResults(result);
      message.destroy();
      message.success('Đồng bộ cơ bản hoàn tất!');
      
      return result;
    } catch (err) {
      console.error('Sync basic error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Có lỗi xảy ra khi đồng bộ cơ bản';
      setError(errorMessage);
      message.destroy();
      message.error(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const syncEnhanced = useCallback(async (owner, repoName) => {
    setLoading(true);
    setError(null);
    setSyncResults(null);

    try {
      message.loading('Đang đồng bộ nâng cao...', 0);
      const result = await syncAPI.syncEnhanced(owner, repoName);
      
      setSyncResults(result);
      message.destroy();
      message.success('Đồng bộ nâng cao hoàn tất!');
      
      return result;
    } catch (err) {
      console.error('Sync enhanced error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Có lỗi xảy ra khi đồng bộ nâng cao';
      setError(errorMessage);
      message.destroy();
      message.error(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const checkGitHubStatus = useCallback(async () => {
    try {
      const result = await syncAPI.checkGitHubStatus();
      return result;
    } catch (err) {
      console.error('GitHub status check error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Không thể kiểm tra trạng thái GitHub API';
      message.error(errorMessage);
      throw err;
    }
  }, []);

  const getRepositoryStats = useCallback(async (owner, repoName) => {
    try {
      const result = await syncAPI.getRepositoryStats(owner, repoName);
      return result;
    } catch (err) {
      console.error('Repository stats error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Không thể lấy thống kê repository';
      message.error(errorMessage);
      throw err;
    }
  }, []);

  const getRepositories = useCallback(async (perPage = 30, page = 1) => {
    try {
      const result = await syncAPI.getRepositories(perPage, page);
      return result;
    } catch (err) {
      console.error('Get repositories error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Không thể lấy danh sách repositories';
      message.error(errorMessage);
      throw err;
    }
  }, []);

  return {
    loading,
    syncResults,
    error,
    syncAll,
    syncBasic,
    syncEnhanced,
    checkGitHubStatus,
    getRepositoryStats,
    getRepositories,
    clearResults
  };
};

export default useSync;
