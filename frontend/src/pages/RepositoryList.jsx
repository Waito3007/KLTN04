import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Typography, 
  Row, 
  Col, 
  Space, 
  Button, 
  Input, 
  Select, 
  Divider,
  Alert,
  Empty,
  Badge
} from 'antd';
import { 
  GithubOutlined, 
  SearchOutlined, 
  SyncOutlined, 
  FilterOutlined,
  AppstoreOutlined,
  BarsOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { Card, Loading, Toast } from '@components/common';
import RepoList from "@components/repo/RepoList";
import axios from 'axios';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;

// Build API URL helper
const buildApiUrl = (endpoint) => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
  return `${baseUrl}${endpoint}`;
};

const RepositoryListPage = () => {
  const navigate = useNavigate();
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('updated_at');
  const [filterBy, setFilterBy] = useState('all');
  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'list'

  // Load repositories từ database
  const loadRepositories = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      Toast.error('Vui lòng đăng nhập lại!');
      navigate('/login');
      return;
    }

    try {
      setLoading(true);
      const response = await axios.get(buildApiUrl('/repositories'), {
        headers: { Authorization: `Bearer ${token}` },
      });
      
      setRepositories(response.data);
      console.log(`Loaded ${response.data.length} repositories from database`);
    } catch (error) {
      console.error('Error loading repositories:', error);
      if (error.response?.status === 401) {
        Toast.error('Phiên đăng nhập hết hạn! Vui lòng đăng nhập lại.');
        navigate('/login');
      } else {
        Toast.error('Không thể tải danh sách repositories!');
      }
    } finally {
      setLoading(false);
    }
  };

  // Sync repositories từ GitHub
  const syncRepositories = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      Toast.error('Vui lòng đăng nhập lại!');
      return;
    }

    try {
      setSyncing(true);
      Toast.info('Đang đồng bộ repositories từ GitHub...');
      
      const response = await axios.post(buildApiUrl('/repositories/sync-all'), {}, {
        headers: { 'Authorization': `Bearer ${token}` },
        timeout: 300000, // 5 minutes timeout
      });

      if (response.data?.status === 'success') {
        Toast.success(`Đồng bộ thành công! Đã đồng bộ ${response.data.synced_count} repositories.`);
        await loadRepositories(); // Reload data after sync
      } else if (response.data?.status === 'partial_success') {
        Toast.warning(`Đồng bộ một phần! Đã đồng bộ ${response.data.synced_count}/${response.data.total_repos} repositories.`);
        
        // Show errors if any
        if (response.data.errors && response.data.errors.length > 0) {
          console.error('Sync errors:', response.data.errors);
          Toast.error(`Có ${response.data.errors.length} lỗi trong quá trình đồng bộ. Kiểm tra console để biết chi tiết.`);
        }
        
        await loadRepositories();
      } else {
        Toast.error(response.data?.message || 'Có lỗi xảy ra khi đồng bộ!');
      }
    } catch (error) {
      console.error('Error syncing repositories:', error);
      
      if (error.response?.status === 401) {
        Toast.error('Phiên đăng nhập hết hạn! Vui lòng đăng nhập lại.');
        navigate('/login');
      } else if (error.code === 'ECONNABORTED') {
        Toast.error('Đồng bộ mất quá nhiều thời gian! Vui lòng thử lại sau.');
      } else if (error.response?.data?.detail) {
        Toast.error(error.response.data.detail);
      } else {
        Toast.error(error.message || 'Có lỗi xảy ra khi đồng bộ repositories!');
      }
    } finally {
      setSyncing(false);
    }
  };

  // Filter và sort repositories
  const getFilteredRepositories = () => {
    let filtered = [...repositories];

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(repo => 
        repo.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        repo.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        repo.owner?.login?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply category filter
    if (filterBy !== 'all') {
      switch (filterBy) {
        case 'public':
          filtered = filtered.filter(repo => !repo.private);
          break;
        case 'private':
          filtered = filtered.filter(repo => repo.private);
          break;
        case 'forked':
          filtered = filtered.filter(repo => repo.fork);
          break;
        case 'original':
          filtered = filtered.filter(repo => !repo.fork);
          break;
        default:
          break;
      }
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'updated_at':
          return new Date(b.updated_at) - new Date(a.updated_at);
        case 'created_at':
          return new Date(b.created_at) - new Date(a.created_at);
        case 'stars':
          return (b.stargazers_count || 0) - (a.stargazers_count || 0);
        default:
          return 0;
      }
    });

    return filtered;
  };

  // Load data on mount
  useEffect(() => {
    loadRepositories();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const filteredRepositories = getFilteredRepositories();

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
        text="Đang tải danh sách repositories..."
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
              <GithubOutlined /> My Repositories
            </Title>
            <Text type="secondary" style={{ fontSize: '16px' }}>
              Quản lý và theo dõi tất cả repositories của bạn
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
                icon={<SyncOutlined />}
                onClick={syncRepositories}
                loading={syncing}
                style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none'
                }}
              >
                {syncing ? 'Đang đồng bộ...' : 'Sync từ GitHub'}
              </Button>
            </Space>
          </Col>
        </Row>

        {/* Stats Overview */}
        <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
          <Col xs={24} sm={8} md={6}>
            <div style={statsCardStyle}>
              <div style={{ fontSize: '28px', fontWeight: 'bold' }}>
                {repositories.length}
              </div>
              <div style={{ fontSize: '14px', opacity: 0.9 }}>
                Total Repositories
              </div>
            </div>
          </Col>
          <Col xs={24} sm={8} md={6}>
            <div style={statsCardStyle}>
              <div style={{ fontSize: '28px', fontWeight: 'bold' }}>
                {repositories.filter(repo => !repo.private).length}
              </div>
              <div style={{ fontSize: '14px', opacity: 0.9 }}>
                Public Repos
              </div>
            </div>
          </Col>
          <Col xs={24} sm={8} md={6}>
            <div style={statsCardStyle}>
              <div style={{ fontSize: '28px', fontWeight: 'bold' }}>
                {repositories.filter(repo => repo.private).length}
              </div>
              <div style={{ fontSize: '14px', opacity: 0.9 }}>
                Private Repos
              </div>
            </div>
          </Col>
          <Col xs={24} sm={8} md={6}>
            <div style={statsCardStyle}>
              <div style={{ fontSize: '28px', fontWeight: 'bold' }}>
                {repositories.reduce((total, repo) => total + (repo.stargazers_count || 0), 0)}
              </div>
              <div style={{ fontSize: '14px', opacity: 0.9 }}>
                Total Stars
              </div>
            </div>
          </Col>
        </Row>
      </div>

      {/* Filters and Search */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} md={8}>
            <Search
              placeholder="Tìm kiếm repositories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              prefix={<SearchOutlined />}
              allowClear
            />
          </Col>
          <Col xs={12} md={4}>
            <Select
              value={filterBy}
              onChange={setFilterBy}
              style={{ width: '100%' }}
              placeholder="Lọc theo"
            >
              <Option value="all">Tất cả</Option>
              <Option value="public">Public</Option>
              <Option value="private">Private</Option>
              <Option value="forked">Forked</Option>
              <Option value="original">Original</Option>
            </Select>
          </Col>
          <Col xs={12} md={4}>
            <Select
              value={sortBy}
              onChange={setSortBy}
              style={{ width: '100%' }}
              placeholder="Sắp xếp"
            >
              <Option value="updated_at">Cập nhật gần đây</Option>
              <Option value="created_at">Tạo mới nhất</Option>
              <Option value="name">Tên A-Z</Option>
              <Option value="stars">Nhiều stars nhất</Option>
            </Select>
          </Col>
          <Col xs={24} md={4}>
            <Button.Group style={{ width: '100%' }}>
              <Button
                type={viewMode === 'grid' ? 'primary' : 'default'}
                icon={<AppstoreOutlined />}
                onClick={() => setViewMode('grid')}
                style={{ width: '50%' }}
              />
              <Button
                type={viewMode === 'list' ? 'primary' : 'default'}
                icon={<BarsOutlined />}
                onClick={() => setViewMode('list')}
                style={{ width: '50%' }}
              />
            </Button.Group>
          </Col>
          <Col xs={24} md={4}>
            <Text type="secondary">
              <Badge count={filteredRepositories.length} style={{ backgroundColor: '#52c41a' }}>
                <FilterOutlined /> Kết quả
              </Badge>
            </Text>
          </Col>
        </Row>
      </Card>

      {/* Repository List */}
      <Card>
        {filteredRepositories.length === 0 ? (
          <Empty
            description={
              searchQuery || filterBy !== 'all' 
                ? "Không tìm thấy repository nào phù hợp với bộ lọc"
                : "Chưa có repository nào"
            }
            style={{ padding: '40px 0' }}
          />
        ) : (
          <RepoList 
            repositories={filteredRepositories}
            onRepoClick={(repo) => navigate(`/repo/${repo.owner?.login}/${repo.name}`)}
          />
        )}
      </Card>

      {/* Help Section */}
      {repositories.length === 0 && (
        <Alert
          type="info"
          showIcon
          message="Bắt đầu với repositories"
          description={
            <div>
              <p>Để bắt đầu, hãy đồng bộ repositories từ GitHub của bạn bằng cách click vào nút "Sync từ GitHub" ở trên.</p>
              <p>Sau khi đồng bộ, bạn có thể xem chi tiết, phân tích commits và quản lý từng repository.</p>
            </div>
          }
          style={{ marginTop: '24px' }}
        />
      )}
    </div>
  );
};

export default RepositoryListPage;
