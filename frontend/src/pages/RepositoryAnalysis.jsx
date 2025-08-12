import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Typography, 
  Row, 
  Col, 
  Space, 
  Button, 
  Select, 
  Alert,
  Empty,
  Badge,
  Tabs,
  Progress,
  Switch,
  Tooltip
} from 'antd';
import { 
  BarChartOutlined, 
  GithubOutlined, 
  ReloadOutlined, 
  SettingOutlined,
  BranchesOutlined,
  FileTextOutlined,
  TeamOutlined,
  CodeOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { Card, Loading } from '@components/common';
import { useToast } from '@components/common/Toast';
import RepoDiagnosisPanel from "@components/Dashboard/components/RepoDiagnosisPanel";
import MemberSkillProfilePanel from "@components/Dashboard/MemberSkill/MemberSkillProfilePanel";
import DashboardAnalyst from "@components/Dashboard/Dashboard_Analyst/DashboardAnalyst";
import axios from 'axios';

const { Title, Text } = Typography;
const { Option } = Select;

// Build API URL helper
const buildApiUrl = (endpoint) => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
  return `${baseUrl}${endpoint}`;
};

const RepositoryAnalysisPage = () => {
  const navigate = useNavigate();
  const toast = useToast();  // Sử dụng hook để tránh context warning
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRepoId, setSelectedRepoId] = useState(null);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [activeTab, setActiveTab] = useState('diagnosis');
  const [branches, setBranches] = useState([]);
  const [branchLoading, setBranchLoading] = useState(false);
  const [daysBack, setDaysBack] = useState(30);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [quickAnalysisMode, setQuickAnalysisMode] = useState(false);
  const dashboardAnalystRef = useRef(null);

  // Load repositories từ database
  const loadRepositories = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      toast.error('Vui lòng đăng nhập lại!');
      navigate('/login');
      return;
    }

    try {
      setLoading(true);
      const response = await axios.get(buildApiUrl('/repositories'), {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      setRepositories(response.data);
      console.log(`Loaded ${response.data.length} repositories for analysis`);
      
      // Auto-select first repository if none selected
      if (response.data.length > 0 && !selectedRepoId) {
        setSelectedRepoId(response.data[0].id);
      }
      
    } catch (error) {
      console.error('Error loading repositories:', error);
      if (error.response?.status === 401) {
        toast.error('Phiên đăng nhập hết hạn! Vui lòng đăng nhập lại.');
        navigate('/login');
      } else {
        toast.error('Không thể tải danh sách repositories!');
      }
    } finally {
      setLoading(false);
    }
  };

  // Load data on mount
  useEffect(() => {
    loadRepositories();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load branches when repository changes
  useEffect(() => {
    if (selectedRepoId) {
      loadBranches();
    } else {
      setBranches([]);
      setSelectedBranch('');
    }
  }, [selectedRepoId, repositories]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load branches for selected repository
  const loadBranches = async () => {
    if (!selectedRepoId) return;

    const token = localStorage.getItem('access_token');
    const selectedRepo = repositories.find(repo => repo.id === selectedRepoId);
    
    if (!selectedRepo || !token) {
      console.warn('❌ No selected repo or token found');
      return;
    }

    // Validate repository data
    if (!selectedRepo.name || (!selectedRepo.owner?.login && !selectedRepo.owner)) {
      console.warn('❌ Invalid repository data:', selectedRepo);
      toast.error('Dữ liệu repository không hợp lệ');
      const fallbackBranches = [
        { value: 'main', label: 'main', isDefault: true },
        { value: 'master', label: 'master', isDefault: false }
      ];
      setBranches(fallbackBranches);
      setSelectedBranch('main');
      return;
    }

    try {
      setBranchLoading(true);
      const ownerName = selectedRepo.owner?.login || selectedRepo.owner || 'Unknown';
      
      console.log(`🌿 Loading branches for: ${ownerName}/${selectedRepo.name}`);
      
      const response = await axios.get(
        buildApiUrl(`/commits/${ownerName}/${selectedRepo.name}/branches`),
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      
      console.log('🌿 Branches API response:', response.data);
      
      // Format branches data
      let branchList = [];
      
      // Handle different response formats
      if (Array.isArray(response.data)) {
        branchList = response.data.map(branch => ({
          value: branch.name || branch,
          label: branch.name || branch,
          isDefault: branch.name === 'main' || branch.name === 'master'
        }));
      } else if (response.data && Array.isArray(response.data.branches)) {
        branchList = response.data.branches.map(branch => ({
          value: branch.name || branch,
          label: branch.name || branch,
          isDefault: branch.name === 'main' || branch.name === 'master'
        }));
      } else {
        console.warn('Unexpected API response format:', response.data);
        throw new Error('Invalid response format');
      }
      
      setBranches(branchList);
      
      // Auto-select default branch
      const defaultBranch = branchList.find(b => b.isDefault) || branchList[0];
      if (defaultBranch && !selectedBranch) {
        setSelectedBranch(defaultBranch.value);
      }
      
      // If no branches found, add warning
      if (branchList.length === 0) {
        console.warn('⚠️ No branches found for repository');
        toast.warning(`Không tìm thấy nhánh nào cho repository ${selectedRepo.name}`);
        const fallbackBranches = [
          { value: 'main', label: 'main (fallback)', isDefault: true }
        ];
        setBranches(fallbackBranches);
        setSelectedBranch('main');
        return;
      }
      
      console.log(`✅ Loaded ${branchList.length} branches for ${selectedRepo.name}`);
      
    } catch (error) {
      console.error('❌ Error loading branches:', error);
      
      // Check specific error types
      if (error.response?.status === 500) {
        console.warn('🚨 Server error - using fallback branches');
        toast.error(`Không thể tải nhánh từ server cho repository ${selectedRepo.name}`);
      } else if (error.response?.status === 404) {
        console.warn('🚨 Repository not found - using fallback branches');
        toast.error(`Repository ${selectedRepo.name} không được tìm thấy`);
      } else {
        console.warn('🚨 Network or other error - using fallback branches');
        toast.error('Lỗi kết nối - sử dụng danh sách nhánh mặc định');
      }
      
      // Fallback to common branch names if API fails
      const fallbackBranches = [
        { value: 'main', label: 'main', isDefault: true },
        { value: 'master', label: 'master', isDefault: false },
        { value: 'develop', label: 'develop', isDefault: false }
      ];
      setBranches(fallbackBranches);
      setSelectedBranch('main');
    } finally {
      setBranchLoading(false);
    }
  };

  // Trigger AI Analysis with retry mechanism
  const triggerAIAnalysis = async (retryCount = 0) => {
    const maxRetries = 2;
    
    if (!selectedRepoId || !selectedBranch) {
      toast.warning('Vui lòng chọn repository và nhánh để phân tích!');
      return;
    }

    const token = localStorage.getItem('access_token');
    if (!token) {
      toast.error('Vui lòng đăng nhập lại!');
      navigate('/login');
      return;
    }

    const selectedRepo = repositories.find(repo => repo.id === selectedRepoId);
    if (!selectedRepo) {
      toast.error('Repository không tồn tại!');
      return;
    }

    setAnalyticsLoading(true);
    setAnalysisProgress(0);
    
    let progressInterval = null;
    
    try {
      const ownerName = selectedRepo.owner?.login || selectedRepo.owner || 'Unknown';
      
      if (retryCount === 0) {
        toast.info('🤖 Đang chạy phân tích AI...');
      } else {
        toast.info(`🔄 Đang thử lại lần ${retryCount}...`);
      }
      
      // Simulate progress với tốc độ chậm hơn cho AI analysis
      progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 85) return prev; // Giữ ở 85% cho đến khi có response
          return prev + Math.random() * 8 + 2; // Tăng chậm hơn
        });
      }, 2000); // Cập nhật mỗi 2 giây
      
      // Call AI analysis API with quick mode support
      const response = await axios.get(
        buildApiUrl(`/dashboard/analytics/${ownerName}/${selectedRepo.name}`),
        {
          headers: { Authorization: `Bearer ${token}` },
          params: { 
            days_back: quickAnalysisMode ? Math.min(daysBack, 7) : daysBack, // Giới hạn 7 ngày cho quick mode
            branch: selectedBranch,
            quick_mode: quickAnalysisMode // Thêm flag cho backend
          },
          timeout: quickAnalysisMode ? 30000 : 120000, // Quick mode: 30s, Normal: 2 phút
          onDownloadProgress: (progressEvent) => {
            // Update progress based on download progress
            if (progressEvent.lengthComputable) {
              const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              setAnalysisProgress(Math.max(percentCompleted, analysisProgress));
            }
          }
        }
      );

      if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
      }
      setAnalysisProgress(100);

      if (response.data) {
        toast.success('✅ Phân tích AI hoàn thành!');
        console.log('AI Analysis results:', response.data);
        
        // Trigger DashboardAnalyst to fetch new data
        if (dashboardAnalystRef.current && dashboardAnalystRef.current.fetchAnalytics) {
          dashboardAnalystRef.current.fetchAnalytics();
        }
        
      } else {
        toast.warning('⚠️ Không có dữ liệu phân tích được trả về.');
      }
      
    } catch (error) {
      console.error('Error during AI analysis:', error);
      
      let errorMessage = '❌ Có lỗi xảy ra khi chạy phân tích AI!';
      
      if (error.response?.status === 401) {
        errorMessage = 'Phiên đăng nhập hết hạn! Vui lòng đăng nhập lại.';
        toast.error(errorMessage);
        navigate('/login');
      } else if (error.response?.status === 404) {
        errorMessage = 'Repository hoặc API endpoint không tồn tại.';
        toast.error(errorMessage);
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Phân tích AI mất quá nhiều thời gian!';
        
        // Retry logic for timeout
        if (retryCount < maxRetries) {
          console.log(`Timeout occurred, retrying... (${retryCount + 1}/${maxRetries})`);
          toast.warning(`⏱️ Timeout - đang thử lại (${retryCount + 1}/${maxRetries})`);
          
          // Clear current states and retry after a short delay
          if (progressInterval) {
            clearInterval(progressInterval);
          }
          setAnalyticsLoading(false);
          setAnalysisProgress(0);
          
          setTimeout(() => {
            triggerAIAnalysis(retryCount + 1);
          }, 2000);
          return;
        } else {
          errorMessage = 'Phân tích AI mất quá nhiều thời gian sau 3 lần thử! Server có thể đang quá tải.';
          
          // Suggest reducing time range if it's large
          if (daysBack > 30) {
            toast.error(errorMessage + ' Hãy thử giảm khoảng thời gian phân tích xuống 30 ngày hoặc ít hơn.');
          } else {
            toast.error(errorMessage + ' Vui lòng thử lại sau hoặc liên hệ admin.');
          }
        }
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
        toast.error(errorMessage);
      } else {
        toast.error(errorMessage);
      }
      
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      setAnalyticsLoading(false);
      setAnalysisProgress(0);
    }
  };

  const selectedRepo = repositories.find(repo => repo.id === selectedRepoId);

  const pageHeaderStyle = {
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%)',
    borderRadius: '16px',
    border: '1px solid rgba(226, 232, 240, 0.3)',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.06)',
    padding: '32px',
    marginBottom: '24px'
  };

  const statsCardStyle = {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    padding: '20px',
    color: 'white',
    textAlign: 'center'
  };

  if (loading) {
    return (
      <Loading 
        variant="gradient"
        text="Đang tải dữ liệu phân tích..."
        size="large"
      />
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* Page Header */}
      <div style={pageHeaderStyle}>
        <Row align="middle" justify="space-between">
          <Col>
            <Title level={2} style={{ margin: 0, color: '#1e293b' }}>
              <BarChartOutlined /> Repository Analysis
            </Title>
            <Text type="secondary" style={{ fontSize: '16px' }}>
              Phân tích chi tiết repositories và thành viên dự án
            </Text>
          </Col>
          <Col>
            <Space>
              <Button
                type="default"
                icon={<ReloadOutlined />}
                onClick={loadRepositories}
                loading={loading}
              >
                Refresh
              </Button>
              <Button
                type="primary"
                icon={<SettingOutlined />}
                onClick={() => navigate('/repo-sync')}
                style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none'
                }}
              >
                Sync Manager
              </Button>
            </Space>
          </Col>
        </Row>

        {/* Repository Selection */}
        <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
          <Col xs={24} md={12}>
            <div>
              <Text strong style={{ display: 'block', marginBottom: '8px' }}>
                Chọn Repository để phân tích:
              </Text>
              <Select
                value={selectedRepoId}
                onChange={setSelectedRepoId}
                style={{ width: '100%' }}
                placeholder="Chọn repository..."
                showSearch
                optionFilterProp="children"
                filterOption={(input, option) =>
                  option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
                }
              >
                {repositories.map(repo => {
                  const ownerName = repo.owner?.login || repo.owner || 'Unknown';
                  return (
                    <Option key={repo.id} value={repo.id}>
                      <Space>
                        <GithubOutlined />
                        <span>
                          <Text type="secondary">{ownerName}/</Text>
                          <Text strong>{repo.name}</Text>
                        </span>
                        {repo.private && <Badge size="small" count="Private" />}
                      </Space>
                    </Option>
                  );
                })}
              </Select>
            </div>
          </Col>
          
          {selectedRepo && (
            <Col xs={24} md={12}>
              <div>
                <Text strong style={{ display: 'block', marginBottom: '8px' }}>
                  Repository được chọn:
                </Text>
                <Card style={{ padding: '12px' }}>
                  <Space direction="vertical" size={4}>
                    <Text strong>{selectedRepo.name}</Text>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {selectedRepo.owner?.login} • {selectedRepo.language || 'N/A'}
                    </Text>
                    {selectedRepo.description && (
                      <Text style={{ fontSize: '12px' }}>
                        {selectedRepo.description}
                      </Text>
                    )}
                  </Space>
                </Card>
              </div>
            </Col>
          )}
        </Row>

        {/* Stats Overview */}
        {selectedRepo && (
          <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
            <Col xs={12} sm={6} md={3}>
              <div style={statsCardStyle}>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {selectedRepo.stargazers_count || 0}
                </div>
                <div style={{ fontSize: '12px', opacity: 0.9 }}>
                  Stars
                </div>
              </div>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <div style={statsCardStyle}>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {selectedRepo.forks_count || 0}
                </div>
                <div style={{ fontSize: '12px', opacity: 0.9 }}>
                  Forks
                </div>
              </div>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <div style={statsCardStyle}>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {selectedRepo.language || 'N/A'}
                </div>
                <div style={{ fontSize: '12px', opacity: 0.9 }}>
                  Language
                </div>
              </div>
            </Col>
            <Col xs={12} sm={6} md={3}>
              <div style={statsCardStyle}>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {selectedRepo.private ? 'Private' : 'Public'}
                </div>
                <div style={{ fontSize: '12px', opacity: 0.9 }}>
                  Visibility
                </div>
              </div>
            </Col>
          </Row>
        )}
      </div>

      {/* No repositories message */}
      {repositories.length === 0 ? (
        <Card>
          <Empty
            description="Chưa có repository nào để phân tích"
            style={{ padding: '40px 0' }}
          >
            <Button 
              type="primary" 
              onClick={() => navigate('/repositories')}
              icon={<GithubOutlined />}
            >
              Đi đến danh sách Repositories
            </Button>
          </Empty>
        </Card>
      ) : !selectedRepoId ? (
        <Alert
          type="info"
          showIcon
          message="Chọn repository để bắt đầu phân tích"
          description="Vui lòng chọn một repository từ dropdown phía trên để xem các công cụ phân tích."
          style={{ marginBottom: '24px' }}
        />
      ) : (
        /* Analysis Tabs */
        <div>
          {/* Repository Info Banner */}
          <Alert
            type="info"
            showIcon
            message={`Đang phân tích repository: ${selectedRepo.name}`}
            description={
              <div>
                <Text>
                  <strong>Owner:</strong> {selectedRepo.owner?.login || selectedRepo.owner} • 
                  <strong> Language:</strong> {selectedRepo.language || 'N/A'} • 
                  <strong> Visibility:</strong> {selectedRepo.private ? 'Private' : 'Public'}
                </Text>
                <br />
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Để thay đổi repository, sử dụng dropdown ở phía trên.
                </Text>
              </div>
            }
            style={{ marginBottom: '16px' }}
          />
          
          <Card>
            <Tabs 
              activeKey={activeTab} 
              onChange={setActiveTab}
              items={[
                {
                  key: 'diagnosis',
                  label: (
                    <span>
                      <FileTextOutlined />
                      Repository Diagnosis
                    </span>
                  ),
                  children: (
                    <RepoDiagnosisPanel 
                      repositories={repositories}
                      selectedRepoId={selectedRepoId}
                      hideRepoSelector={true}
                      onBranchChange={setSelectedBranch}
                    />
                  ),
                },
                {
                  key: 'ai-analysis',
                  label: (
                    <span>
                      <BarChartOutlined />
                      AI Analysis
                    </span>
                  ),
                  children: (
                    <div style={{ padding: '0' }}>
                      {/* AI Analysis Header Controls */}
                      <Card 
                        style={{ 
                          marginBottom: '24px',
                          background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)',
                          borderColor: 'rgba(102, 126, 234, 0.2)'
                        }}
                      >
                        <Row gutter={[16, 16]} align="middle">
                          <Col xs={24} sm={12} md={6}>
                            <div>
                              <Text strong style={{ display: 'block', marginBottom: '8px', fontSize: '13px' }}>
                                <BranchesOutlined style={{ color: '#667eea' }} /> Chọn nhánh:
                              </Text>
                              <Select
                                style={{ width: '100%' }}
                                placeholder={branchLoading ? "Đang tải nhánh..." : "Chọn nhánh để phân tích"}
                                value={selectedBranch}
                                onChange={setSelectedBranch}
                                disabled={!selectedRepoId || branchLoading}
                                loading={branchLoading}
                                size="middle"
                                notFoundContent={branchLoading ? "Đang tải..." : "Không có nhánh"}
                                showSearch
                              >
                                {branches.map(branch => (
                                  <Option key={branch.value} value={branch.value}>
                                    <Space>
                                      <BranchesOutlined style={{ fontSize: '12px' }} />
                                      <span>{branch.label}</span>
                                      {branch.isDefault && (
                                        <Badge size="small" count="default" style={{ backgroundColor: '#52c41a' }} />
                                      )}
                                    </Space>
                                  </Option>
                                ))}
                              </Select>
                            </div>
                          </Col>
                          <Col xs={24} sm={12} md={6}>
                            <div>
                              <Text strong style={{ display: 'block', marginBottom: '8px', fontSize: '13px' }}>
                                <ClockCircleOutlined style={{ color: '#667eea' }} /> Khoảng thời gian:
                              </Text>
                              <Select 
                                value={daysBack} 
                                onChange={setDaysBack}
                                style={{ width: '100%' }} 
                                size="middle"
                              >
                                <Option value={7}>7 ngày qua</Option>
                                <Option value={30}>30 ngày qua</Option>
                                <Option value={90}>90 ngày qua</Option>
                                <Option value={180}>6 tháng qua</Option>
                              </Select>
                            </div>
                          </Col>
                          <Col xs={24} sm={24} md={12}>
                            <div style={{ display: 'flex', alignItems: 'end', height: '100%', paddingTop: '20px' }}>
                              {/* Quick Analysis Toggle */}
                              <div style={{ marginBottom: '12px', width: '100%' }}>
                                <Tooltip title="Phân tích nhanh sử dụng ít dữ liệu hơn và hoàn thành trong 30 giây">
                                  <div style={{ marginBottom: '8px' }}>
                                    <Switch
                                      checked={quickAnalysisMode}
                                      onChange={setQuickAnalysisMode}
                                      size="small"
                                    />
                                    <Text style={{ marginLeft: '8px', fontSize: '12px' }}>
                                      ⚡ Quick Analysis {quickAnalysisMode ? '(≤7 ngày, 30s)' : '(Tắt)'}
                                    </Text>
                                  </div>
                                </Tooltip>
                                
                                <Button
                                  type="primary"
                                  icon={<BarChartOutlined />}
                                  size="large"
                                  loading={analyticsLoading}
                                  onClick={() => triggerAIAnalysis(0)}
                                  style={{ 
                                    width: '100%',
                                    height: '44px',
                                    background: analyticsLoading 
                                      ? 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)'
                                      : quickAnalysisMode
                                      ? 'linear-gradient(135deg, #ff7875 0%, #ffa940 100%)'
                                      : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                    border: 'none',
                                    borderRadius: '8px',
                                    fontWeight: 500,
                                    fontSize: '15px',
                                    boxShadow: analyticsLoading
                                      ? '0 4px 15px rgba(82, 196, 26, 0.3)'
                                      : quickAnalysisMode
                                      ? '0 4px 15px rgba(255, 120, 117, 0.3)'
                                      : '0 4px 15px rgba(102, 126, 234, 0.3)'
                                  }}
                                  disabled={!selectedRepoId || !selectedBranch}
                                >
                                  {analyticsLoading 
                                    ? '🔄 Đang phân tích...' 
                                    : quickAnalysisMode 
                                    ? '⚡ Phân tích nhanh' 
                                    : '🤖 Chạy phân tích AI'
                                  }
                                </Button>
                              </div>
                            </div>
                          </Col>
                        </Row>
                      </Card>

                      {/* AI Analysis Status Banner */}
                      {analyticsLoading && (
                        <Alert
                          type="info"
                          showIcon
                          message={`🤖 Đang chạy phân tích AI${quickAnalysisMode ? ' (Chế độ nhanh)' : ''}...`}
                          description={
                            <div>
                              <Text>
                                Hệ thống AI đang phân tích repository <strong>{selectedRepo.name}</strong> trên nhánh <strong>{selectedBranch}</strong>
                                {quickAnalysisMode && (
                                  <Badge 
                                    count="⚡ NHANH" 
                                    style={{ backgroundColor: '#ff7875', marginLeft: '8px' }}
                                  />
                                )}
                              </Text>
                              <br />
                              <div style={{ margin: '12px 0 8px 0' }}>
                                <Progress 
                                  percent={Math.round(analysisProgress)} 
                                  strokeColor={
                                    quickAnalysisMode ? {
                                      '0%': '#ff7875',
                                      '100%': '#ffa940',
                                    } : {
                                      '0%': '#108ee9',
                                      '100%': '#87d068',
                                    }
                                  }
                                  showInfo={true}
                                  format={(percent) => `${percent}%`}
                                />
                              </div>
                              <Text type="secondary" style={{ fontSize: '12px' }}>
                                {quickAnalysisMode ? (
                                  analysisProgress < 50 ? 'Phân tích nhanh - Thu thập dữ liệu cơ bản...' :
                                  'Phân tích nhanh - Tạo insights...'
                                ) : (
                                  analysisProgress < 30 ? 'Đang thu thập dữ liệu commit...' :
                                  analysisProgress < 60 ? 'Đang phân tích patterns...' :
                                  analysisProgress < 90 ? 'Đang tạo insights...' : 
                                  'Hoàn thiện kết quả...'
                                )}
                                {quickAnalysisMode && (
                                  <span style={{ color: '#ff7875', fontWeight: 500 }}>
                                    {' '}(≤30s)
                                  </span>
                                )}
                              </Text>
                            </div>
                          }
                          style={{ marginBottom: '16px' }}
                          banner
                        />
                      )}

                      {/* Full-width AI Analysis Content */}
                      <div style={{ minHeight: '400px' }}>
                        <DashboardAnalyst 
                          ref={dashboardAnalystRef}
                          selectedRepoId={selectedRepoId}
                          repositories={repositories}
                          onBranchChange={setSelectedBranch}
                          fullWidth={true}
                        />
                      </div>
                    </div>
                  ),
                },
                {
                  key: 'member-skills',
                  label: (
                    <span>
                      <TeamOutlined />
                      Member Skills
                    </span>
                  ),
                  children: (
                    <MemberSkillProfilePanel 
                      repositories={repositories}
                      selectedRepoId={selectedRepoId}
                      selectedBranch={selectedBranch}
                      hideRepoSelector={true}
                    />
                  ),
                },
                {
                  key: 'code-analysis',
                  label: (
                    <span>
                      <CodeOutlined />
                      Code Analysis
                    </span>
                  ),
                  children: (
                    <div style={{ padding: '40px', textAlign: 'center' }}>
                      <Empty
                        description="Code Analysis sẽ được phát triển trong phiên bản tiếp theo"
                        image={<CodeOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />}
                      />
                    </div>
                  ),
                },
              ]}
            />
          </Card>
        </div>
      )}

      {/* Help Section */}
      {repositories.length > 0 && (
        <Alert
          type="info"
          showIcon
          message="Hướng dẫn sử dụng"
          description={
            <div>
              <p><strong>Repository Diagnosis:</strong> Phân tích tổng quan về repository, branches và commits.</p>
              <p><strong>AI Analysis:</strong> Sử dụng AI để phân tích patterns và insights từ commit history.</p>
              <p><strong>Member Skills:</strong> Phân tích kỹ năng và đóng góp của từng thành viên trong team.</p>
              <p>Chọn repository khác nhau để so sánh và phân tích đa dạng dự án.</p>
            </div>
          }
          style={{ marginTop: '24px' }}
        />
      )}
    </div>
  );
};

export default RepositoryAnalysisPage;
