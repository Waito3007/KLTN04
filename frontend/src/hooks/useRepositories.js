/**
 * useRepositories - Custom hook cho quản lý repositories
 * Tuân thủ quy tắc KLTN04: Defensive programming, validation, caching
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { message } from 'antd';
import { repositoryAPI } from '../services/repositoryAPI';

const useRepositories = (autoLoad = true) => {
  // State management
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedRepository, setSelectedRepository] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Error handler
  const handleError = useCallback((error, action = 'tải repositories') => {
    console.error(`Lỗi khi ${action}:`, error);
    
    let errorMessage = `Không thể ${action}`;
    
    if (error.response?.status === 401) {
      errorMessage = 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.';
    } else if (error.response?.status === 403) {
      errorMessage = 'Bạn không có quyền truy cập vào repositories này.';
    } else if (error.response?.status === 404) {
      errorMessage = 'Không tìm thấy repositories.';
    } else if (error.response?.data?.message) {
      errorMessage = error.response.data.message;
    } else if (error.message) {
      errorMessage = error.message;
    }
    
    setError(errorMessage);
    message.error(errorMessage);
  }, []);

  // Load repositories từ API
  const loadRepositories = useCallback(async (showLoading = true) => {
    try {
      if (showLoading) {
        setLoading(true);
      }
      setError(null);

      const response = await repositoryAPI.getRepositories();
      
      // Validation response
      if (!response || !Array.isArray(response.data)) {
        throw new Error('Dữ liệu repositories không hợp lệ');
      }

      // Validate repository structure
      const validRepositories = response.data.filter(repo => {
        return repo && 
               typeof repo.id === 'number' && 
               typeof repo.name === 'string' && 
               typeof repo.owner === 'string';
      });

      if (validRepositories.length !== response.data.length) {
        console.warn('Một số repositories có cấu trúc không hợp lệ và đã bị loại bỏ');
      }

      setRepositories(validRepositories);
      
      // Auto-select first repository if none selected
      if (!selectedRepository && validRepositories.length > 0) {
        setSelectedRepository(validRepositories[0]);
      }
      
    } catch (error) {
      handleError(error, 'tải danh sách repositories');
      setRepositories([]);
    } finally {
      setLoading(false);
    }
  }, [selectedRepository, handleError]);

  // Filter repositories theo search term
  const filteredRepositories = useMemo(() => {
    if (!searchTerm) return repositories;
    
    const searchLower = searchTerm.toLowerCase().trim();
    if (!searchLower) return repositories;

    return repositories.filter(repo => {
      return (
        repo.name?.toLowerCase().includes(searchLower) ||
        repo.owner?.toLowerCase().includes(searchLower) ||
        repo.full_name?.toLowerCase().includes(searchLower) ||
        repo.description?.toLowerCase().includes(searchLower)
      );
    });
  }, [repositories, searchTerm]);

  // Repository options cho Select component
  const repositoryOptions = useMemo(() => {
    return filteredRepositories.map(repo => ({
      value: repo.id,
      label: `${repo.owner}/${repo.name}`,
      repo: repo,
      key: repo.id
    }));
  }, [filteredRepositories]);

  // Select repository
  const selectRepository = useCallback((repositoryId) => {
    if (!repositoryId) {
      setSelectedRepository(null);
      return;
    }

    const repository = repositories.find(repo => repo.id === repositoryId);
    if (!repository) {
      message.warning('Repository không tồn tại');
      return;
    }

    setSelectedRepository(repository);
  }, [repositories]);

  // Get repository by ID
  const getRepositoryById = useCallback((repositoryId) => {
    if (!repositoryId) return null;
    return repositories.find(repo => repo.id === repositoryId) || null;
  }, [repositories]);

  // Get repository by name
  const getRepositoryByName = useCallback((owner, name) => {
    if (!owner || !name) return null;
    return repositories.find(repo => 
      repo.owner === owner && repo.name === name
    ) || null;
  }, [repositories]);

  // Search repositories
  const searchRepositories = useCallback((term) => {
    setSearchTerm(term || '');
  }, []);

  // Clear search
  const clearSearch = useCallback(() => {
    setSearchTerm('');
  }, []);

  // Refresh repositories
  const refreshRepositories = useCallback(() => {
    loadRepositories(true);
  }, [loadRepositories]);

  // Repository statistics
  const repositoryStats = useMemo(() => {
    return {
      total: repositories.length,
      public: repositories.filter(repo => !repo.private).length,
      private: repositories.filter(repo => repo.private).length,
      filtered: filteredRepositories.length
    };
  }, [repositories, filteredRepositories]);

  // Validate repository selection
  const validateRepository = useCallback((repository) => {
    if (!repository) {
      throw new Error('Repository không được để trống');
    }

    if (!repository.id) {
      throw new Error('Repository ID không hợp lệ');
    }

    if (!repository.name || !repository.owner) {
      throw new Error('Thông tin repository không đầy đủ');
    }

    return true;
  }, []);

  // Get repository full name
  const getRepositoryFullName = useCallback((repository) => {
    if (!repository) return '';
    return `${repository.owner}/${repository.name}`;
  }, []);

  // Check if repository is selected
  const isRepositorySelected = useCallback((repositoryId) => {
    return selectedRepository?.id === repositoryId;
  }, [selectedRepository]);

  // Load repositories khi component mount
  useEffect(() => {
    if (autoLoad) {
      loadRepositories();
    }
  }, [autoLoad, loadRepositories]);

  // Update selected repository khi repositories thay đổi
  useEffect(() => {
    if (selectedRepository && repositories.length > 0) {
      const updatedRepo = repositories.find(repo => repo.id === selectedRepository.id);
      if (updatedRepo && updatedRepo !== selectedRepository) {
        setSelectedRepository(updatedRepo);
      } else if (!updatedRepo) {
        // Repository đã bị xóa hoặc không có quyền truy cập
        setSelectedRepository(null);
      }
    }
  }, [repositories, selectedRepository]);

  return {
    // Data
    repositories: filteredRepositories,
    allRepositories: repositories,
    selectedRepository,
    repositoryOptions,
    repositoryStats,
    
    // State
    loading,
    error,
    searchTerm,
    
    // Actions
    loadRepositories,
    refreshRepositories,
    selectRepository,
    searchRepositories,
    clearSearch,
    
    // Getters
    getRepositoryById,
    getRepositoryByName,
    getRepositoryFullName,
    
    // Validators
    validateRepository,
    isRepositorySelected
  };
};

export default useRepositories;
