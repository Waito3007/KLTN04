/**
 * Hằng số cho localStorage keys
 */
export const STORAGE_KEYS = {
  GITHUB_PROFILE: 'github_profile',
  USER: 'user',
  ACCESS_TOKEN: 'access_token'
};

/**
 * Hằng số cho routes
 */
export const ROUTES = {
  PUBLIC: {
    HOME: '/',
    LOGIN: '/login',
    AUTH_SUCCESS: '/auth-success'
  },
  PROTECTED: {
    DASHBOARD: '/dashboard',
    REPOSITORIES: '/repositories',
    ANALYSIS: '/analysis',
    DEMO: '/demo',
    SYNC: '/sync',
    REPO_SYNC: '/repo-sync',
    REPO_DETAILS: '/repo-details',
    COMMITS: '/commits',
    TEST: '/test'
  }
};

/**
 * Hằng số cho thông báo
 */
export const MESSAGES = {
  AUTH: {
    LOGIN_SUCCESS: 'Đăng nhập thành công!',
    LOGOUT_SUCCESS: 'Đăng xuất thành công!',
    LOGIN_ERROR: 'Có lỗi xảy ra khi đăng nhập!',
    LOGOUT_ERROR: 'Có lỗi xảy ra khi đăng xuất!',
    UNAUTHORIZED: 'Bạn cần đăng nhập để truy cập trang này!',
    INVALID_USER_DATA: 'Dữ liệu người dùng không hợp lệ',
    STORAGE_ERROR: 'Không thể lưu thông tin đăng nhập',
    UPDATE_ERROR: 'Không thể cập nhật thông tin người dùng'
  },
  LOADING: {
    CHECKING_AUTH: 'Đang kiểm tra trạng thái đăng nhập...',
    LOADING: 'Đang tải...'
  }
};

/**
 * Hằng số cho cấu hình
 */
export const CONFIG = {
  AUTH_CHECK_INTERVAL: 30000, // 30 giây
  TOKEN_REFRESH_THRESHOLD: 300000, // 5 phút
  MAX_RETRY_ATTEMPTS: 3
};
