import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Card,
  Button,
  Table,
  Tag,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  Tabs,
  Tooltip,
  Progress,
  Modal,
  Select,
  Avatar,
  Badge,
  List,
  Timeline,
  Drawer,
  notification
} from 'antd';
import { Toast, Loading } from '@components/common';
import {
  SyncOutlined,
  GithubOutlined,
  StarFilled,
  ForkOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  HistoryOutlined,
  BellOutlined
} from '@ant-design/icons';
import styled from 'styled-components';
import { repoSyncAPI } from "@services/api";

const { Title, Text } = Typography;
const { Option } = Select;

const Container = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
`;

const StatsCard = styled(Card)`
  text-align: center;
  border-radius: 8px;
  .ant-statistic-content-value {
    color: ${props => props.color || '#1890ff'};
  }
`;

const RepoCard = styled(Card)`
  margin-bottom: 16px;
  border-radius: 8px;
  transition: all 0.3s;
  
  &:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const PriorityBadge = styled(Badge)`
  .ant-badge-count {
    background-color: ${props => {
      switch (props.priority) {
        case 'highest': return '#ff4d4f';
        case 'high': return '#faad14';
        case 'normal': return '#52c41a';
        default: return '#d9d9d9';
      }
    }};
  }
`;

const RepoSyncManager = () => {
  const [loading, setLoading] = useState(false);
  const [syncStatus, setSyncStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('unsynced');
  const [syncing, setSyncing] = useState({});
  const [syncModalVisible, setSyncModalVisible] = useState(false);
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [syncType, setSyncType] = useState('optimized');
  const [eventsDrawerVisible, setEventsDrawerVisible] = useState(false);
  const [selectedRepoEvents, setSelectedRepoEvents] = useState(null);
  const wsRef = useRef(null);
  
  // New progress states
  const [globalSyncProgress, setGlobalSyncProgress] = useState({
    visible: false,
    totalRepos: 0,
    completedRepos: 0,
    currentRepo: '',
    overallProgress: 0
  });
  const [repoProgresses, setRepoProgresses] = useState({}); // Track individual repo progress

  const handleSyncEvent = useCallback((event) => {
    const { repo_key, event_type, data } = event;
    
    // Update sync events
    // Update sync events
    // (Removed setSyncEvents as syncEvents state is unused)
    // Real-time update sync status based on event
    const [owner, repoName] = repo_key.split('/');
    
    setSyncStatus(prevStatus => {
      if (!prevStatus) return prevStatus;
      
      // Find the repo in all categories and update/move it
      let targetRepo = null;
      let sourceCategory = null;
      
      // Find which category the repo is currently in
      ['unsynced', 'outdated', 'synced'].forEach(category => {
        const categoryKey = category === 'unsynced' ? 'unsynced' : 
                           category === 'outdated' ? 'outdated' : 'synced';
        const repos = prevStatus.repositories?.[categoryKey] || [];
        const found = repos.find(repo => repo.owner === owner && repo.name === repoName);
        if (found) {
          targetRepo = { ...found };
          sourceCategory = categoryKey;
        }
      });
      
      if (!targetRepo) return prevStatus; // Repo not found
      
      // Update repo based on event type
      switch (event_type) {
        case 'sync_start':
          targetRepo.sync_status = 'syncing';
          targetRepo.sync_progress = 0;
          
          // Update individual repo progress
          setRepoProgresses(prev => ({
            ...prev,
            [repo_key]: { progress: 0, stage: 'Bắt đầu đồng bộ...', status: 'syncing' }
          }));
          break;
          
        case 'sync_progress':
          targetRepo.sync_status = 'syncing';
          targetRepo.sync_progress = data.percentage || 0;
          targetRepo.sync_stage = data.stage || '';
          
          // Update individual repo progress
          setRepoProgresses(prev => ({
            ...prev,
            [repo_key]: { 
              progress: data.percentage || 0, 
              stage: data.stage || 'Đang xử lý...', 
              status: 'syncing' 
            }
          }));
          break;
          
        case 'sync_complete':
          if (data.success) {
            targetRepo.sync_status = 'sync_completed';
            targetRepo.sync_progress = 100;
            targetRepo.last_synced = new Date().toISOString();
            targetRepo.sync_stage = 'Completed';
            targetRepo.needs_initial_sync = false;
            targetRepo.is_outdated = false; // Explicitly set to false after successful sync
            
            // Update individual repo progress
            setRepoProgresses(prev => ({
              ...prev,
              [repo_key]: { progress: 100, stage: 'Hoàn thành', status: 'completed' }
            }));
            
            // Auto-clear completed status after 5 seconds
            setTimeout(() => {
              setRepoProgresses(prev => {
                const newState = { ...prev };
                delete newState[repo_key];
                return newState;
              });
            }, 5000);
          } else {
            targetRepo.sync_status = 'sync_failed';
            targetRepo.sync_progress = 0;
            targetRepo.sync_stage = 'Failed';
            
            // Update individual repo progress
            setRepoProgresses(prev => ({
              ...prev,
              [repo_key]: { progress: 0, stage: 'Thất bại', status: 'failed' }
            }));
            
            // Auto-clear failed status after 10 seconds
            setTimeout(() => {
              setRepoProgresses(prev => {
                const newState = { ...prev };
                delete newState[repo_key];
                return newState;
              });
            }, 10000);
          }
          break;
          
        case 'sync_error':
          targetRepo.sync_status = 'sync_failed';
          targetRepo.sync_progress = 0;
          targetRepo.sync_stage = `Error: ${data.error}`;
          
          // Update individual repo progress
          setRepoProgresses(prev => ({
            ...prev,
            [repo_key]: { progress: 0, stage: `Lỗi: ${data.error}`, status: 'failed' }
          }));
          
          // Auto-clear error status after 10 seconds
          setTimeout(() => {
            setRepoProgresses(prev => {
              const newState = { ...prev };
              delete newState[repo_key];
              return newState;
            });
          }, 10000);
          break;
      }
      
      // Create new repositories object
      const newRepositories = {
        unsynced: [...(prevStatus.repositories.unsynced || [])],
        outdated: [...(prevStatus.repositories.outdated || [])],
        synced: [...(prevStatus.repositories.synced || [])]
      };
      
      // Remove repo from source category
      if (sourceCategory) {
        newRepositories[sourceCategory] = newRepositories[sourceCategory].filter(
          repo => !(repo.owner === owner && repo.name === repoName)
        );
      }
      
      // Determine target category and add repo
      let targetCategory = sourceCategory; // Default to keeping in same category
      
      if (event_type === 'sync_complete' && data.success) {
        // Move successfully synced repos to 'synced' category
        targetCategory = 'synced';
      } else if (event_type === 'sync_error') {
        // Keep failed syncs in their original category (unsynced/outdated)
        if (sourceCategory === 'synced') {
          targetCategory = 'outdated'; // If a synced repo fails, move to outdated
        }
      }
      // For sync_start and sync_progress, keep in original category
      
      // Add updated repo to target category
      newRepositories[targetCategory].push(targetRepo);
      
      // Update summary counts
      const newSummary = {
        ...prevStatus.summary,
        unsynced_count: newRepositories.unsynced.length,
        outdated_count: newRepositories.outdated.length,
        synced_count: newRepositories.synced.length
      };
      
      return {
        ...prevStatus,
        repositories: newRepositories,
        summary: newSummary
      };
    });

    // Show notifications
    switch (event_type) {
      case 'sync_start':
        notification.info({
          message: 'Đồng bộ bắt đầu',
          description: `Bắt đầu đồng bộ ${repo_key} (${data.sync_type})`,
          placement: 'topRight',
          duration: 3
        });
        setSyncing(prev => ({ ...prev, [repo_key]: true }));
        break;
        
      case 'sync_progress':
        // Progress updates - no notification to avoid spam
        break;
        
      case 'sync_complete':
        if (data.success) {
          notification.success({
            message: 'Đồng bộ thành công',
            description: `Repository ${repo_key} đã được đồng bộ thành công và chuyển sang danh sách "Đã đồng bộ"`,
            placement: 'topRight',
            duration: 4
          });
        } else {
          notification.error({
            message: 'Đồng bộ thất bại',
            description: `Repository ${repo_key} đồng bộ thất bại`,
            placement: 'topRight',
            duration: 4
          });
        }
        setSyncing(prev => ({ ...prev, [repo_key]: false }));
        // Don't call fetchSyncStatus() since we're updating in real-time
        break;
        
      case 'sync_error':
        notification.error({
          message: 'Lỗi đồng bộ',
          description: `Lỗi khi đồng bộ ${repo_key}: ${data.error}`,
          placement: 'topRight',
          duration: 6
        });
        setSyncing(prev => ({ ...prev, [repo_key]: false }));
        break;
    }
  }, []);

  const setupPollingFallback = useCallback(() => {
    console.log('📊 Setting up polling fallback');
    
    // Kiểm tra token trước khi setup polling
    const token = localStorage.getItem('access_token');
    if (!token) {
      console.log('⚠️ No access token found, skipping polling setup');
      wsRef.current = { 
        close: () => console.log('No-op polling stopped'),
        readyState: 1,
        isPolling: true
      };
      return;
    }
    
    notification.info({
      message: 'Chế độ offline',
      description: 'Không thể kết nối server. Sử dụng chế độ offline - sự kiện đồng bộ sẽ không được cập nhật real-time.',
      placement: 'topRight',
      duration: 5
    });

    // Store mock polling for cleanup
    wsRef.current = { 
      close: () => console.log('Mock polling stopped'),
      readyState: 1, // Simulate connected state
      isPolling: true
    };
  }, []);

  const connectWebSocket = useCallback(() => {
    // Kiểm tra token trước khi kết nối WebSocket
    const token = localStorage.getItem('access_token');
    if (!token) {
      console.log('⚠️ No access token found, skipping WebSocket connection');
      setupPollingFallback();
      return;
    }

    // First, try to connect with WebSocket with better error handling
    console.log('🔌 Attempting WebSocket connection...');
    
    try {
      // Close existing connection if any
      if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
        wsRef.current.close();
      }

      const ws = new WebSocket('ws://localhost:8000/api/sync-events/ws');
      wsRef.current = ws;

      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          console.log('⏰ WebSocket connection timeout, falling back to polling');
          ws.close();
          setupPollingFallback();
        }
      }, 5000); // 5 second timeout

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('✅ WebSocket connected successfully');
        
        // Send initial ping
        ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
        
        notification.success({
          message: 'Kết nối real-time thành công',
          description: 'Đã kết nối WebSocket. Sự kiện đồng bộ sẽ được cập nhật tức thời.',
          placement: 'topRight',
          duration: 3
        });
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', data);
          
          if (data.type === 'connection_established') {
            console.log('🎉 WebSocket connection established');
          } else if (data.type === 'echo') {
            console.log('🔄 WebSocket echo response:', data);
          } else if (data.repo_key && data.event_type) {
            // This is a sync event
            handleSyncEvent(data);
          }
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        
        // Only reconnect if it wasn't a clean close and we haven't fallen back to polling
        if (event.code !== 1000 && !wsRef.current?.isPolling) {
          console.log('🔄 Attempting to reconnect WebSocket in 3 seconds...');
          setTimeout(() => {
            if (!wsRef.current?.isPolling) {
              connectWebSocket();
            }
          }, 3000);
        }
      };

      ws.onerror = (error) => {
        clearTimeout(connectionTimeout);
        console.error('❌ WebSocket error:', error);
        
        notification.warning({
          message: 'Chuyển sang chế độ polling',
          description: 'WebSocket không thể kết nối. Đang chuyển sang chế độ polling.',
          placement: 'topRight',
          duration: 4
        });
        
        // Fallback to polling
        setupPollingFallback();
      };

    } catch (error) {
      console.error('❌ Failed to create WebSocket:', error);
      setupPollingFallback();
    }
  }, [handleSyncEvent, setupPollingFallback]);

  useEffect(() => {
    // Initialize component - start with data fetch
    fetchSyncStatus();
    
    // Use a small delay to ensure component is fully mounted before WebSocket
    const connectTimer = setTimeout(() => {
      connectWebSocket();
    }, 500); // 500ms delay to avoid initialization issues
    
    return () => {
      clearTimeout(connectTimer);
      if (wsRef.current) {
        wsRef.current.close();
      }
      // Cleanup progress states
      setGlobalSyncProgress(prev => ({ ...prev, visible: false }));
      setRepoProgresses({});
    };
  }, [connectWebSocket]); // Add connectWebSocket to dependencies

  const fetchSyncStatus = async () => {
    // Kiểm tra token trước khi gọi API
    const token = localStorage.getItem('access_token');
    if (!token) {
      console.log('⚠️ No access token found, skipping sync status fetch');
      setSyncStatus({
        summary: {
          total_github_repos: 0,
          unsynced_count: 0,
          outdated_count: 0,
          synced_count: 0
        },
        repositories: {
          unsynced: [],
          outdated: [],
          synced: []
        }
      });
      return;
    }

    setLoading(true);
    try {
      const data = await repoSyncAPI.getSyncStatus();
      
      // Debug: Log the data structure to understand categorization
      console.log('=== SYNC STATUS DEBUG ===');
      console.log('Raw data from backend:', data);
      
      if (data.repositories) {
        console.log('Repositories categorization:');
        Object.keys(data.repositories).forEach(category => {
          const repos = data.repositories[category];
          console.log(`${category}: ${repos.length} repos`);
          
          // Log first few repos in each category for debugging
          repos.slice(0, 3).forEach(repo => {
            console.log(`  - ${repo.name}: needs_initial_sync=${repo.needs_initial_sync}, is_outdated=${repo.is_outdated}, last_synced=${repo.last_synced}`);
          });
        });
      }
      
      setSyncStatus(data);
    } catch (error) {
      console.error('Fetch sync status error:', error);
      
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        Toast.error('Server không phản hồi. Vui lòng kiểm tra backend server.');
      } else if (error.response?.status === 401) {
        Toast.error('Lỗi xác thực. Vui lòng đăng nhập lại.');
      } else {
        Toast.error('Lỗi khi tải danh sách repositories');
      }
      
      // Set empty data to prevent UI crash
      setSyncStatus({
        summary: {
          total_github_repos: 0,
          total_db_repos: 0,
          unsynced_count: 0,
          outdated_count: 0,
          synced_count: 0
        },
        repositories: {
          unsynced: [],
          outdated: [],
          synced: []
        },
        user: null
      });
    } finally {
      setLoading(false);
    }
  };

  const showRepoEvents = async (repo) => {
    try {
      const data = await repoSyncAPI.getRepoEvents(repo.owner, repo.name);
      setSelectedRepoEvents(data);
      setEventsDrawerVisible(true);
    } catch (err) {
      Toast.error('Lỗi khi tải sự kiện đồng bộ');
      console.error('Get repo events error:', err);
    }
  };

  const handleSyncRepository = async (repo, type = 'optimized') => {
    const repoKey = `${repo.owner}/${repo.name}`;
    
    // Check if we have sync status data (server is reachable)
    if (!syncStatus || !syncStatus.summary) {
      Toast.warning('Server không khả dụng. Vui lòng kiểm tra kết nối backend.');
      return;
    }
    
    setSyncing(prev => ({ ...prev, [repoKey]: true }));
    
    try {
      await repoSyncAPI.syncRepository(repo.owner, repo.name, type);
      
      if (type === 'optimized') {
        Toast.success(`Đã bắt đầu đồng bộ background cho ${repo.full_name || repo.name}`);
      } else {
        Toast.success(`Đồng bộ ${repo.full_name || repo.name} thành công`);
      }
      
      // No need to call fetchSyncStatus() anymore since we have real-time updates
      
    } catch (error) {
      let errorMessage = `Lỗi đồng bộ ${repo.full_name || repo.name}`;
      
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorMessage += ': Server không phản hồi';
      } else if (error.response?.data?.detail) {
        errorMessage += `: ${error.response.data.detail}`;
      } else {
        errorMessage += `: ${error.message}`;
      }
      
      Toast.error(errorMessage);
      console.error('Sync error:', error);
      setSyncing(prev => ({ ...prev, [repoKey]: false }));
    }
  };

  const handleSyncAllRepos = async () => {
    if (!syncStatus?.repositories) {
      Toast.warning('Không có dữ liệu repositories để đồng bộ');
      return;
    }

    const allRepos = [
      ...(syncStatus.repositories.unsynced || []),
      ...(syncStatus.repositories.outdated || [])
    ];

    if (allRepos.length === 0) {
      Toast.info('Tất cả repositories đã được đồng bộ');
      return;
    }

    // Show global progress
    setGlobalSyncProgress({
      visible: true,
      totalRepos: allRepos.length,
      completedRepos: 0,
      currentRepo: 'Chuẩn bị đồng bộ...',
      overallProgress: 0
    });

    try {
      let completedCount = 0;
      
      // Start syncing all repos with a small delay to avoid overwhelming the server
      for (let i = 0; i < allRepos.length; i++) {
        const repo = allRepos[i];
        const repoKey = `${repo.owner}/${repo.name}`;
        
        // Update current repo being synced
        setGlobalSyncProgress(prev => ({
          ...prev,
          currentRepo: repoKey
        }));
        
        try {
          await handleSyncRepository(repo, 'optimized');
          
          completedCount++;
          
          // Update global progress
          setGlobalSyncProgress(prev => ({
            ...prev,
            completedRepos: completedCount,
            overallProgress: (completedCount / allRepos.length) * 100
          }));
          
          // Small delay between sync starts
          if (i < allRepos.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 1000)); // 1s delay
          }
        } catch (error) {
          console.error(`Error syncing ${repo.name}:`, error);
          completedCount++; // Still count as completed to continue
        }
      }

      Toast.info(`Đã bắt đầu đồng bộ ${allRepos.length} repositories`);
      
      // Hide global progress after a delay
      setTimeout(() => {
        setGlobalSyncProgress(prev => ({ ...prev, visible: false }));
      }, 3000);
      
    } catch (error) {
      console.error('Sync all repos error:', error);
      Toast.error('Lỗi khi đồng bộ tất cả repositories');
      setGlobalSyncProgress(prev => ({ ...prev, visible: false }));
    }
  };

  const openSyncModal = (repo) => {
    setSelectedRepo(repo);
    setSyncModalVisible(true);
  };

  const handleModalSync = () => {
    if (selectedRepo) {
      handleSyncRepository(selectedRepo, syncType);
      setSyncModalVisible(false);
      setSelectedRepo(null);
    }
  };

  const getSyncStatusTag = (repo) => {
    // Debug log for each repo
    console.log(`Tag for ${repo.name}:`, {
      sync_status: repo.sync_status,
      needs_initial_sync: repo.needs_initial_sync,
      is_outdated: repo.is_outdated,
      last_synced: repo.last_synced
    });
    
    // Check real-time sync status first
    switch (repo.sync_status) {
      case 'syncing':
        return <Tag color="blue" icon={<SyncOutlined spin />}>Đang đồng bộ</Tag>;
      case 'sync_completed':
        return <Tag color="green" icon={<CheckCircleOutlined />}>Đã đồng bộ</Tag>;
      case 'sync_failed':
        return <Tag color="red" icon={<ExclamationCircleOutlined />}>Lỗi đồng bộ</Tag>;
    }
    
    // Fix logic: Check needs_initial_sync explicitly
    if (repo.needs_initial_sync === true) {
      return <Tag color="red">Chưa đồng bộ</Tag>;
    }
    
    // Fix logic: If needs_initial_sync is false or undefined, check is_outdated
    if (repo.is_outdated === true) {
      return <Tag color="orange">Cần cập nhật</Tag>;
    }
    
    // Fix logic: If needs_initial_sync is not true and is_outdated is not true, it's synced
    if (repo.needs_initial_sync !== true && repo.is_outdated !== true) {
      return <Tag color="green">Đã đồng bộ</Tag>;
    }
    
    // Default fallback
    return <Tag color="gray">Không xác định</Tag>;
  };

  const getColumns = () => [
    {
      title: 'Repository',
      key: 'repo',
      render: (_, repo) => (
        <Space>
          <Avatar 
            size="small" 
            src={`https://github.com/${repo.owner}.png`}
            icon={<GithubOutlined />}
          />
          <div>
            <Text strong>{repo.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {repo.owner}
            </Text>
          </div>
        </Space>
      ),
      width: 200,
    },
    {
      title: 'Mô tả',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (text) => text || <Text type="secondary">Không có mô tả</Text>
    },
    {
      title: 'Ngôn ngữ',
      dataIndex: 'language',
      key: 'language',
      render: (language) => language ? <Tag color="blue">{language}</Tag> : '-',
      width: 100,
    },
    {
      title: 'Thống kê',
      key: 'stats',
      render: (_, repo) => (
        <Space>
          <Tooltip title="Stars">
            <Space size={4}>
              <StarFilled style={{ color: '#faad14' }} />
              <Text>{repo.stars || 0}</Text>
            </Space>
          </Tooltip>
          <Tooltip title="Forks">
            <Space size={4}>
              <ForkOutlined style={{ color: '#1890ff' }} />
              <Text>{repo.forks || 0}</Text>
            </Space>
          </Tooltip>
        </Space>
      ),
      width: 120,
    },
    {
      title: 'Trạng thái',
      key: 'status',
      render: (_, repo) => {
        const repoKey = `${repo.owner}/${repo.name}`;
        const repoProgress = repoProgresses[repoKey];
        
        return (
          <Space direction="vertical" size={4}>
            {getSyncStatusTag(repo)}
            {repo.sync_priority && (
              <PriorityBadge 
                priority={repo.sync_priority}
                count={repo.sync_priority === 'highest' ? 'HOT' : repo.sync_priority === 'high' ? '!' : ''}
                size="small"
              />
            )}
            
            {/* Enhanced sync progress display */}
            {(repo.sync_status === 'syncing' || repoProgress?.status === 'syncing') && (
              <div style={{ width: '120px' }}>
                <Progress 
                  percent={repoProgress?.progress || repo.sync_progress || 0} 
                  size="small" 
                  status="active"
                  format={(percent) => `${percent}%`}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
                {(repoProgress?.stage || repo.sync_stage) && (
                  <Text style={{ fontSize: '10px', color: '#666', display: 'block', marginTop: 2 }}>
                    {repoProgress?.stage || repo.sync_stage}
                  </Text>
                )}
              </div>
            )}
            
            {/* Show completed progress briefly */}
            {repoProgress?.status === 'completed' && (
              <div style={{ width: '120px' }}>
                <Progress 
                  percent={100} 
                  size="small" 
                  status="success"
                  format={() => '✓'}
                />
                <Text style={{ fontSize: '10px', color: '#52c41a', display: 'block', marginTop: 2 }}>
                  {repoProgress.stage}
                </Text>
              </div>
            )}
            
            {/* Show failed progress */}
            {repoProgress?.status === 'failed' && (
              <div style={{ width: '120px' }}>
                <Progress 
                  percent={0} 
                  size="small" 
                  status="exception"
                  format={() => '✗'}
                />
                <Text style={{ fontSize: '10px', color: '#ff4d4f', display: 'block', marginTop: 2 }}>
                  {repoProgress.stage}
                </Text>
              </div>
            )}
            
            {/* Debug info - remove after fixing */}
            <div style={{ fontSize: '10px', color: '#999' }}>
              <div>needs_sync: {String(repo.needs_initial_sync)}</div>
              <div>outdated: {String(repo.is_outdated)}</div>
            </div>
          </Space>
        );
      },
      width: 180,
    },
    {
      title: 'Cập nhật',
      key: 'updated',
      render: (_, repo) => (
        <Space direction="vertical" size={4}>
          {repo.github_updated_at && (
            <Tooltip title={`GitHub: ${new Date(repo.github_updated_at).toLocaleString()}`}>
              <Space size={4}>
                <ClockCircleOutlined />
                <Text style={{ fontSize: '12px' }}>
                  {new Date(repo.github_updated_at).toLocaleDateString()}
                </Text>
              </Space>
            </Tooltip>
          )}
          {repo.last_synced && (
            <Tooltip title={`Đồng bộ: ${new Date(repo.last_synced).toLocaleString()}`}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                DB: {new Date(repo.last_synced).toLocaleDateString()}
              </Text>
            </Tooltip>
          )}
        </Space>
      ),
      width: 120,
    },
    {
      title: 'Thao tác',
      key: 'actions',
      render: (_, repo) => {
        const repoKey = `${repo.owner}/${repo.name}`;
        return (
          <Space>
            <Button
              type="primary"
              size="small"
              icon={<SyncOutlined />}
              loading={syncing[repoKey]}
              onClick={() => handleSyncRepository(repo)}
            >
              Đồng bộ
            </Button>
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => openSyncModal(repo)}
            >
              Tùy chọn
            </Button>
            <Button
              size="small"
              icon={<HistoryOutlined />}
              onClick={() => showRepoEvents(repo)}
            >
              Sự kiện
            </Button>
          </Space>
        );
      },
      width: 200,
    },
  ];

  const renderSummary = () => {
    if (!syncStatus) return null;

    const { summary } = syncStatus;
    
    return (
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <StatsCard color="#1890ff">
            <Statistic 
              title="Tổng GitHub Repos" 
              value={summary.total_github_repos}
              prefix={<GithubOutlined />}
            />
          </StatsCard>
        </Col>
        <Col span={6}>
          <StatsCard color="#ff4d4f">
            <Statistic 
              title="Chưa đồng bộ" 
              value={summary.unsynced_count}
              prefix={<ExclamationCircleOutlined />}
            />
          </StatsCard>
        </Col>
        <Col span={6}>
          <StatsCard color="#faad14">
            <Statistic 
              title="Cần cập nhật" 
              value={summary.outdated_count}
              prefix={<ClockCircleOutlined />}
            />
          </StatsCard>
        </Col>
        <Col span={6}>
          <StatsCard color="#52c41a">
            <Statistic 
              title="Đã đồng bộ" 
              value={summary.synced_count}
              prefix={<CheckCircleOutlined />}
            />
          </StatsCard>
        </Col>
      </Row>
    );
  };

  const getTabCount = (type) => {
    if (!syncStatus) return 0;
    return syncStatus.repositories[type]?.length || 0;
  };

  const getTabItems = () => [
    {
      key: 'unsynced',
      label: (
        <Badge count={getTabCount('unsynced')} offset={[10, 0]}>
          <span>Chưa đồng bộ</span>
        </Badge>
      ),
      children: (
        <Table
          columns={getColumns()}
          dataSource={syncStatus?.repositories?.unsynced || []}
          rowKey={(record) => `${record.owner}-${record.name}`}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      ),
    },
    {
      key: 'outdated',
      label: (
        <Badge count={getTabCount('outdated')} offset={[10, 0]}>
          <span>Cần cập nhật</span>
        </Badge>
      ),
      children: (
        <Table
          columns={getColumns()}
          dataSource={syncStatus?.repositories?.outdated || []}
          rowKey={(record) => `${record.owner}-${record.name}`}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      ),
    },
    {
      key: 'synced',
      label: (
        <Badge count={getTabCount('synced')} offset={[10, 0]}>
          <span>Đã đồng bộ</span>
        </Badge>
      ),
      children: (
        <Table
          columns={getColumns()}
          dataSource={syncStatus?.repositories?.synced || []}
          rowKey={(record) => `${record.owner}-${record.name}`}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      ),
    },
  ];

  if (loading && !syncStatus) {
    return (
      <Container>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Loading variant="circle" size="large" message="Đang tải danh sách repositories..." />
        </div>
      </Container>
    );
  }

  return (
    <Container>
      <Row justify="space-between" align="middle" style={{ marginBottom: 24 }}>
        <Col>
          <Title level={2}>
            <GithubOutlined /> Repository Sync Manager
          </Title>
          <Text type="secondary">
            Quản lý đồng bộ repositories từ GitHub
          </Text>
        </Col>
        <Col>
          <Space>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={fetchSyncStatus}
              loading={loading}
            >
              Làm mới
            </Button>
            <Button 
              type="primary"
              icon={<SyncOutlined />} 
              onClick={handleSyncAllRepos}
              loading={Object.keys(syncing).length > 0}
            >
              Đồng bộ tất cả
            </Button>
          </Space>
        </Col>
      </Row>

      {renderSummary()}

      {/* Global Sync Progress */}
      {globalSyncProgress.visible && (
        <Card style={{ marginBottom: 24, backgroundColor: '#f6ffed', border: '1px solid #b7eb8f' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Text strong style={{ color: '#389e0d' }}>
                <SyncOutlined spin /> Đang đồng bộ tất cả repositories
              </Text>
              <Text style={{ color: '#389e0d' }}>
                {globalSyncProgress.completedRepos}/{globalSyncProgress.totalRepos}
              </Text>
            </div>
            
            <Progress 
              percent={globalSyncProgress.overallProgress} 
              status="active"
              strokeColor={{
                '0%': '#52c41a',
                '100%': '#389e0d',
              }}
              format={(percent) => `${Math.round(percent)}%`}
            />
            
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Hiện tại: {globalSyncProgress.currentRepo}
            </Text>
          </Space>
        </Card>
      )}

      <Card>
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          items={getTabItems()}
        />
      </Card>

      <Modal
        title="Tùy chọn đồng bộ"
        open={syncModalVisible}
        onOk={handleModalSync}
        onCancel={() => setSyncModalVisible(false)}
        okText="Đồng bộ"
        cancelText="Hủy"
      >
        {selectedRepo && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Repository: </Text>
              <Text>{selectedRepo.full_name || selectedRepo.name}</Text>
            </div>
            
            <div>
              <Text strong>Loại đồng bộ: </Text>
              <Select 
                value={syncType} 
                onChange={setSyncType}
                style={{ width: '100%', marginTop: 8 }}
              >
                <Option value="basic">
                  <Space>
                    <SyncOutlined />
                    <span>Cơ bản (Repository + Branches)</span>
                  </Space>
                </Option>
                <Option value="enhanced">
                  <Space>
                    <SyncOutlined />
                    <span>Nâng cao (+ Commits + Issues + PRs)</span>
                  </Space>
                </Option>
                <Option value="optimized">
                  <Space>
                    <SyncOutlined />
                    <span>Tối ưu (Background + Concurrent + Diff)</span>
                  </Space>
                </Option>
              </Select>
            </div>

            <div style={{ marginTop: 16 }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {syncType === 'basic' && 'Đồng bộ thông tin repository và branches cơ bản'}
                {syncType === 'enhanced' && 'Đồng bộ đầy đủ commits, issues và pull requests'}
                {syncType === 'optimized' && 'Đồng bộ background với tốc độ cao nhất, bao gồm code diff'}
              </Text>
            </div>
          </Space>
        )}
      </Modal>

      {/* Events Drawer */}
      <Drawer
        title={
          <Space>
            <HistoryOutlined />
            <span>Sự kiện đồng bộ</span>
            {selectedRepoEvents && (
              <Tag color="blue">{selectedRepoEvents.repo_key}</Tag>
            )}
          </Space>
        }
        placement="right"
        width={500}
        open={eventsDrawerVisible}
        onClose={() => setEventsDrawerVisible(false)}
      >
        {selectedRepoEvents && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>Tổng số sự kiện: </Text>
              <Badge count={selectedRepoEvents.total_events} />
            </div>
            
            <Timeline mode="left">
              {selectedRepoEvents.events.map((event, index) => {
                const getEventIcon = (eventType) => {
                  switch (eventType) {
                    case 'sync_start':
                      return <SyncOutlined style={{ color: '#1890ff' }} />;
                    case 'sync_progress':
                      return <ClockCircleOutlined style={{ color: '#faad14' }} />;
                    case 'sync_complete':
                      return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
                    case 'sync_error':
                      return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
                    default:
                      return <BellOutlined />;
                  }
                };

                const getEventColor = (eventType) => {
                  switch (eventType) {
                    case 'sync_start': return 'blue';
                    case 'sync_progress': return 'orange';
                    case 'sync_complete': return 'green';
                    case 'sync_error': return 'red';
                    default: return 'gray';
                  }
                };

                return (
                  <Timeline.Item
                    key={index}
                    dot={getEventIcon(event.event_type)}
                    color={getEventColor(event.event_type)}
                  >
                    <div>
                      <div style={{ marginBottom: 4 }}>
                        <Tag color={getEventColor(event.event_type)}>
                          {event.event_type.replace('_', ' ').toUpperCase()}
                        </Tag>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(event.timestamp).toLocaleString()}
                        </Text>
                      </div>
                      
                      {event.data && (
                        <div style={{ marginTop: 8 }}>
                          {event.event_type === 'sync_start' && (
                            <Text>Bắt đầu đồng bộ ({event.data.sync_type})</Text>
                          )}
                          {event.event_type === 'sync_progress' && (
                            <div>
                              <Progress 
                                percent={event.data.percentage} 
                                size="small" 
                                status="active"
                              />
                              <Text style={{ fontSize: '12px' }}>
                                {event.data.current}/{event.data.total} - {event.data.stage}
                              </Text>
                            </div>
                          )}
                          {event.event_type === 'sync_complete' && (
                            <Text type="success">Đồng bộ hoàn thành thành công</Text>
                          )}
                          {event.event_type === 'sync_error' && (
                            <Text type="danger">{event.data.error}</Text>
                          )}
                        </div>
                      )}
                    </div>
                  </Timeline.Item>
                );
              })}
            </Timeline>
          </div>
        )}
      </Drawer>
    </Container>
  );
};

export default RepoSyncManager;
