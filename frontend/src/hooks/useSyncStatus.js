// e:\Project\KLTN04-1\frontend\src\hooks\useSyncStatus.js
import { useState, useEffect } from 'react';
import { getSyncStatus } from '@services/syncService';

const useSyncStatus = () => {
  const [syncStatus, setSyncStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchSyncStatus = async () => {
    setLoading(true);
    try {
      const response = await getSyncStatus();

      // Kiểm tra Content-Type
      if (!response.headers || !response.headers['content-type']?.includes('application/json')) {
        throw new Error('API trả về dữ liệu không phải JSON');
      }

      const data = await response.json();
      if (typeof data !== 'object' || data === null) {
        throw new Error('Dữ liệu JSON không hợp lệ');
      }

      console.log('Dữ liệu trả về từ API:', data);
      setSyncStatus(data);
    } catch (error) {
      console.error('Lỗi khi gọi API /api/sync-status:', error);
      if (error.response) {
        console.error('Chi tiết response:', {
          status: error.response.status,
          statusText: error.response.statusText,
          body: await error.response.text(),
        });
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSyncStatus();
  }, []);

  return { syncStatus, loading, fetchSyncStatus };
};

export default useSyncStatus;
