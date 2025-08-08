import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export const aiService = {
  /**
   * Calls the backend to classify a single commit using the MultiFusion model.
   * @param {number} commitId The ID of the commit to classify.
   * @returns {Promise<{predicted_class: string, confidence: number}>}
   */
  classifyCommitWithMultiFusion: async (commitId) => {
    if (!commitId) {
      throw new Error('Commit ID is required');
    }
    try {
      const response = await apiClient.post(`/ai/commits/${commitId}/classify`);
      return response.data;
    } catch (error) {      console.error(`Error classifying commit ${commitId} with MultiFusion:`, error);
      // Trả về một cấu trúc lỗi để component có thể xử lý
      return { error: true, message: error.response?.data?.detail || 'Lỗi không xác định từ server' };
    }
  },

  // Bạn có thể thêm các hàm gọi API AI khác ở đây trong tương lai
  // ví dụ: getHanAnalysis(repoId, memberLogin, branchName)
};
