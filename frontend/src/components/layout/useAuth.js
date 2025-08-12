import { useContext } from 'react';
import { AuthContext } from '@contexts/AuthContext';
import { MESSAGES } from '@constants/auth';

/**
 * Custom hook để sử dụng AuthContext
 * @returns {Object} Auth context value bao gồm user, isLoading, isAuthenticated, login, logout, updateUser
 * @throws {Error} Nếu được sử dụng bên ngoài AuthProvider
 */
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth phải được sử dụng bên trong AuthProvider");
  }
  return context;
};

// Export default để tiện import
export default useAuth;