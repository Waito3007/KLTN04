import { useEffect, useState, useCallback } from "react";
import { Select, Spin, message, Tag, Typography, Button, Space, Tooltip } from "antd";
import { GithubOutlined, BranchesOutlined, SyncOutlined, DatabaseOutlined } from '@ant-design/icons';
import axios from "axios";
import styled from "styled-components";

const { Option } = Select;
const { Text } = Typography;

const SelectContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  flex-wrap: wrap;

  @media (max-width: 768px) {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
  }
`;

const BranchControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;

  @media (max-width: 768px) {
    flex-direction: column;
    align-items: stretch;
  }
`;

const SyncControls = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
`;

const SyncButton = styled(Button)`
  display: flex;
  align-items: center;
  gap: 4px;
  height: 32px;
  padding: 0 12px;
  
  &.ant-btn-primary {
    background: linear-gradient(135deg, #1890ff 0%, #096dd9 100%);
    border: none;
    
    &:hover {
      background: linear-gradient(135deg, #40a9ff 0%, #1890ff 100%);
      transform: translateY(-1px);
    }
  }
  
  &.ant-btn-default {
    border-color: #52c41a;
    color: #52c41a;
    
    &:hover {
      border-color: #73d13d;
      color: #73d13d;
      background: #f6ffed;
    }
  }
`;

const StyledSelect = styled(Select)`
  min-width: 240px;
  
  .ant-select-selector {
    border-radius: 6px !important;
    border: 1px solid #d9d9d9 !important;
    transition: all 0.3s !important;
    
    &:hover {
      border-color: #1890ff !important;
    }
  }
  
  .ant-select-selection-item {
    font-weight: 500;
  }
`;

const BranchTag = styled(Tag)`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 6px;
  background: #f0f5ff;
  color: #1890ff;
  border: 1px solid #d6e4ff;
`;

const BranchSelector = ({ owner, repo, onBranchChange }) => {
  const [branches, setBranches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedBranch, setSelectedBranch] = useState(null);
  const [syncLoading, setSyncLoading] = useState(false);
  const [commitStats, setCommitStats] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false); // Prevent multiple initializations

  // Define fetchCommitStats first using useCallback
  const fetchCommitStats = useCallback(async (branchName) => {
    const token = localStorage.getItem("access_token");
    if (!token) return;

    try {
      const response = await axios.get(
        `http://localhost:8000/api/commits/${owner}/${repo}/branches/${branchName}/commits?limit=1`,
        {
          headers: {
            Authorization: `token ${token}`,
          },
        }
      );
      setCommitStats({
        totalCommits: response.data.total_found || 0,
        lastSync: new Date().toLocaleString()
      });
    } catch (err) {
      console.error("Error fetching commit stats:", err);
    }
  }, [owner, repo]);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) return;

    const fetchBranches = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8000/api/github/${owner}/${repo}/branches`,
          {
            headers: {
              Authorization: `token ${token}`,
            },
          }
        );        setBranches(response.data);
        if (response.data.length > 0 && !isInitialized) {
          const defaultBranch = response.data[0].name;
          console.log('BranchSelector: Initializing with default branch:', defaultBranch);
          setSelectedBranch(defaultBranch);
          onBranchChange(defaultBranch);
          fetchCommitStats(defaultBranch);
          setIsInitialized(true);
        }
      } catch (err) {
        console.error(err);
        message.error("Không lấy được danh sách branch");
      } finally {
        setLoading(false);
      }
    };    fetchBranches();
  }, [owner, repo, onBranchChange, fetchCommitStats, isInitialized]);  const handleBranchChange = (value) => {
    console.log('BranchSelector: Changing branch from', selectedBranch, 'to', value);
    console.log('Available branches:', branches);
    
    if (value === selectedBranch) {
      console.log('BranchSelector: Same branch selected, skipping');
      return;
    }
    
    setSelectedBranch(value);
    onBranchChange(value);
    fetchCommitStats(value);
  };

  const syncCommitsForBranch = async () => {
    if (!selectedBranch) {
      message.warning("Vui lòng chọn branch trước!");
      return;
    }

    const token = localStorage.getItem("access_token");
    if (!token) {
      message.error("Vui lòng đăng nhập lại!");
      return;
    }

    setSyncLoading(true);
    try {
      const response = await axios.post(
        `http://localhost:8000/api/github/${owner}/${repo}/branches/${selectedBranch}/sync-commits?include_stats=true&per_page=100&max_pages=5`,
        {},
        {
          headers: {
            Authorization: `token ${token}`,
          },
        }
      );
      
      const { stats } = response.data;
      message.success(
        `Đồng bộ thành công! ${stats.commits_processed} commits được xử lý cho branch "${selectedBranch}"`
      );
      
      // Update commit stats
      setCommitStats({
        totalCommits: stats.total_commits_in_database,
        newCommits: stats.commits_processed,
        lastSync: new Date().toLocaleString()
      });
      
      // Auto refresh commit stats after sync
      setTimeout(() => {
        fetchCommitStats(selectedBranch);
      }, 1000);
      
    } catch (error) {
      console.error("Lỗi khi đồng bộ commits:", error);
      const errorMessage = error.response?.data?.detail || "Không thể đồng bộ commits!";
      message.error(errorMessage);
    } finally {
      setSyncLoading(false);
    }
  };

  const viewCommitsInDB = async () => {
    if (!selectedBranch) {
      message.warning("Vui lòng chọn branch trước!");
      return;
    }

    const token = localStorage.getItem("access_token");
    if (!token) {
      message.error("Vui lòng đăng nhập lại!");
      return;
    }

    try {
      const response = await axios.get(
        `http://localhost:8000/api/commits/${owner}/${repo}/branches/${selectedBranch}/commits?limit=10`,
        {
          headers: {
            Authorization: `token ${token}`,
          },
        }
      );
      
      const { commits, count } = response.data;
      if (count > 0) {
        message.success(`Tìm thấy ${count} commits trong database cho branch "${selectedBranch}"`);
        console.log("Commits in database:", commits);
        
        // Trigger parent refresh if callback available
        if (onBranchChange) {
          onBranchChange(selectedBranch, { refresh: true });
        }
      } else {
        message.info(`Chưa có commits nào trong database cho branch "${selectedBranch}". Hãy đồng bộ trước!`);
      }
      
    } catch (error) {
      console.error("Lỗi khi xem commits:", error);
      message.error("Không thể lấy danh sách commits!");
    }
  };

  if (loading) {
    console.log('BranchSelector: Loading...');
    return <Spin size="small" />;
  }

  if (!branches || branches.length === 0) {
    console.log('BranchSelector: No branches available');
  } else {
    console.log('BranchSelector: Available branches:', branches.map(b => b.name));
  }

  console.log('BranchSelector: Current selected branch:', selectedBranch);

  return (
    <div style={{ marginBottom: 16 }}>
      <SelectContainer>
        <BranchControls>
          <BranchTag>
            <BranchesOutlined />
            <Text strong>Branch:</Text>
          </BranchTag>
            <StyledSelect
            value={selectedBranch}
            onChange={handleBranchChange}
            suffixIcon={<GithubOutlined style={{ color: '#1890ff' }} />}
            popupMatchSelectWidth={false}
          >
            {branches.map((branch) => (
              <Option key={branch.name} value={branch.name}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <BranchesOutlined style={{ color: '#52c41a' }} />
                  <Text strong>{branch.name}</Text>
                </div>
              </Option>
            ))}
          </StyledSelect>
        </BranchControls>
        
        <SyncControls>
          <Tooltip title="Đồng bộ commits từ GitHub cho branch này">
            <SyncButton
              type="primary"
              size="small"
              loading={syncLoading}
              onClick={syncCommitsForBranch}
              disabled={!selectedBranch}
            >
              <SyncOutlined />
              Sync
            </SyncButton>
          </Tooltip>
          
          <Tooltip title="Xem commits đã lưu trong database">
            <SyncButton
              type="default"
              size="small"
              onClick={viewCommitsInDB}
              disabled={!selectedBranch}
            >
              <DatabaseOutlined />
              View DB
            </SyncButton>
          </Tooltip>
        </SyncControls>
      </SelectContainer>
      
      {commitStats && (
        <div style={{ 
          marginTop: 8, 
          padding: '6px 12px', 
          background: '#f0f5ff', 
          borderRadius: '6px',
          fontSize: '12px',
          color: '#1890ff'
        }}>
          <Space split={<span style={{ color: '#d9d9d9' }}>|</span>}>
            <span>📊 {commitStats.totalCommits} commits</span>
            {commitStats.newCommits && (
              <span>✨ {commitStats.newCommits} mới</span>
            )}
            <span>🕒 {commitStats.lastSync}</span>
          </Space>
        </div>
      )}
    </div>
  );
};

export default BranchSelector;
