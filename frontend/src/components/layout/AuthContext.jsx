import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { AuthContext } from '@contexts/AuthContext';
import { STORAGE_KEYS, MESSAGES } from '@constants/auth';

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Hàm tiện ích để xử lý localStorage một cách an toàn
  const getStoredUserData = useCallback(() => {
    try {
      const storedData = localStorage.getItem(STORAGE_KEYS.GITHUB_PROFILE) || 
                        localStorage.getItem(STORAGE_KEYS.USER);
      return storedData ? JSON.parse(storedData) : null;
    } catch (error) {
      console.error('Lỗi khi đọc dữ liệu người dùng từ localStorage:', error);
      // Clear invalid data
      localStorage.removeItem(STORAGE_KEYS.GITHUB_PROFILE);
      localStorage.removeItem(STORAGE_KEYS.USER);
      return null;
    }
  }, []);

  // Khởi tạo trạng thái người dùng từ localStorage
  useEffect(() => {
    const initializeAuth = () => {
      setIsLoading(true);
      
      const userData = getStoredUserData();
      
      if (userData) {
        setUser(userData);
        setIsAuthenticated(true);
      } else {
        setUser(null);
        setIsAuthenticated(false);
      }
      
      setIsLoading(false);
    };

    initializeAuth();
  }, [getStoredUserData]);

  // Hàm đăng nhập
  const login = useCallback((userData) => {
    if (!userData) {
      console.error(MESSAGES.AUTH.INVALID_USER_DATA);
      return;
    }

    try {
      setUser(userData);
      setIsAuthenticated(true);
      localStorage.setItem(STORAGE_KEYS.GITHUB_PROFILE, JSON.stringify(userData));
    } catch (error) {
      console.error('Lỗi khi lưu dữ liệu người dùng:', error);
      throw new Error(MESSAGES.AUTH.STORAGE_ERROR);
    }
  }, []);

  // Hàm đăng xuất
  const logout = useCallback(() => {
    try {
      setUser(null);
      setIsAuthenticated(false);
      
      // Clear tất cả dữ liệu liên quan đến xác thực
      Object.values(STORAGE_KEYS).forEach(key => {
        localStorage.removeItem(key);
      });
    } catch (error) {
      console.error('Lỗi khi đăng xuất:', error);
    }
  }, []);

  // Hàm cập nhật thông tin người dùng
  const updateUser = useCallback((updatedData) => {
    if (!updatedData) {
      console.error(MESSAGES.AUTH.INVALID_USER_DATA);
      return;
    }

    try {
      const newUserData = { ...user, ...updatedData };
      setUser(newUserData);
      localStorage.setItem(STORAGE_KEYS.GITHUB_PROFILE, JSON.stringify(newUserData));
    } catch (error) {
      console.error('Lỗi khi cập nhật thông tin người dùng:', error);
      throw new Error(MESSAGES.AUTH.UPDATE_ERROR);
    }
  }, [user]);

  // Memoize context value để tránh re-render không cần thiết
  const contextValue = useMemo(() => ({
    user,
    isLoading,
    isAuthenticated,
    login,
    logout,
    updateUser
  }), [user, isLoading, isAuthenticated, login, logout, updateUser]);

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};