// frontend/src/pages/SyncPage.jsx
import React, { useState, useEffect } from 'react';
import { Row, Col, Input, Button, Space, Typography, Alert } from 'antd';
import { GithubOutlined, DatabaseOutlined } from '@ant-design/icons';
import RepositorySync from '@components/RepositorySync';
import { useSync } from "@hooks/useSync";
import Navbar from "@components/common/Navbar";
import { Card as CustomCard, Button as CustomButton } from "@components/common";

const { Title, Text } = Typography;
const { Search } = Input; // Lưu ý: Search được import nhưng không được sử dụng.

const SyncPage = () => {
  const [owner, setOwner] = useState('Waito3007');
  const [repoName, setRepoName] = useState('KLTN04');
  const [showSync, setShowSync] = useState(false);
  const [gitHubStatus, setGitHubStatus] = useState(null);
  const [checkingStatus, setCheckingStatus] = useState(false);

  const { checkGitHubStatus } = useSync();

  // Placeholder user data
  const user = { username: 'User', avatar_url: '', email: 'user@example.com' };
  // Placeholder syncing state
  const isSyncing = false;
  // Placeholder sync function
  const syncAllRepositories = () => console.log('Syncing repositories...');
  // Placeholder logout function
  const handleLogout = () => console.log('Logging out...');

  // Check GitHub status on component mount
  useEffect(() => {
    const checkStatus = async () => {
      setCheckingStatus(true);
      try {
        const status = await checkGitHubStatus();
        setGitHubStatus(status);
      } catch (error) {
        console.error('Error checking GitHub status:', error);
      } finally {
        setCheckingStatus(false);
      }
    };
    
    checkStatus();
  }, [checkGitHubStatus]);

  const handleCheckGitHubStatus = async () => {
    setCheckingStatus(true);
    try {
      const status = await checkGitHubStatus();
      setGitHubStatus(status);
    } catch (error) {
      console.error('Error checking GitHub status:', error);
    } finally {
      setCheckingStatus(false);
    }
  };

  const handleStartSync = () => {
    if (owner && repoName) {
      setShowSync(true);
    }
  };

  const handleSyncComplete = (result) => {
    console.log('Sync completed:', result);
    // Có thể thực hiện các hành động khác sau khi sync hoàn tất
  };

  const renderGitHubStatus = () => {
    if (checkingStatus) {
      return (
        <CustomCard title="Trạng thái GitHub API" loading>
          <Text>Đang kiểm tra...</Text>
        </CustomCard>
      );
    }

    if (!gitHubStatus) return null;

    const { github_api_accessible, token_valid, rate_limit, token_provided } = gitHubStatus;

    return (
      <CustomCard 
        title={
          <Space>
            <GithubOutlined />
            <span>Trạng thái GitHub API</span>
          </Space>
        }
        extra={
          <CustomButton size="small" onClick={handleCheckGitHubStatus}>
            Kiểm tra lại
          </CustomButton>
        }
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>GitHub API: </Text>
            {github_api_accessible ? (
              <Text type="success">✅ Kết nối thành công</Text>
            ) : (
              <Text type="danger">❌ Không thể kết nối</Text>
            )}
          </div>
          
          <div>
            <Text strong>Token: </Text>
            {token_provided ? (
              token_valid ? (
                <Text type="success">✅ Hợp lệ</Text>
              ) : (
                <Text type="danger">❌ Không hợp lệ</Text>
              )
            ) : (
              <Text type="warning">⚠️ Chưa cung cấp</Text>
            )}
          </div>

          {rate_limit && (
            <div>
              <Text strong>Rate Limit: </Text>
              <Text>
                {rate_limit.remaining}/{rate_limit.limit} requests còn lại
              </Text>
              {rate_limit.remaining < 100 && (
                <Text type="warning"> (Sắp đạt giới hạn)</Text>
              )}
            </div>
          )}
        </Space>
      </CustomCard>
    );
  };

  return (
    <div style={{ padding: '24px', minHeight: '100vh' }}>
      <Navbar 
        user={user} 
        isSyncing={isSyncing} 
        syncAllRepositories={syncAllRepositories} 
        handleLogout={handleLogout} 
      />
      <Row justify="center">
        <Col xs={24} sm={20} md={16} lg={12} xl={10}>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            
            {/* Header */}
            <CustomCard>
              <div style={{ textAlign: 'center' }}>
                <Title level={2}>
                  <DatabaseOutlined /> Repository Sync Tool
                </Title>
                <Text type="secondary">
                  Công cụ đồng bộ dữ liệu repository từ GitHub vào hệ thống
                </Text>
              </div>
            </CustomCard>

            {/* GitHub Status */}
            {renderGitHubStatus()}

            {/* Repository Input */}
            <CustomCard title="Chọn Repository">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>Owner:</Text>
                  <Input
                    placeholder="Tên owner (VD: Waito3007)"
                    value={owner}
                    onChange={(e) => setOwner(e.target.value)}
                    style={{ marginTop: 8 }}
                  />
                </div>
                
                <div>
                  <Text strong>Repository:</Text>
                  <Input
                    placeholder="Tên repository (VD: KLTN04)"
                    value={repoName}
                    onChange={(e) => setRepoName(e.target.value)}
                    style={{ marginTop: 8 }}
                    onPressEnter={handleStartSync}
                  />
                </div>

                <Button
                  type="primary"
                  block
                  onClick={handleStartSync}
                  disabled={!owner || !repoName}
                  icon={<GithubOutlined />}
                >
                  Chọn Repository
                </Button>
              </Space>
            </CustomCard>

            {/* Sync Component */}
            {showSync && (
              <RepositorySync 
                owner={owner}
                repoName={repoName}
                onSyncComplete={handleSyncComplete}
              />
            )}

            {/* Instructions */}
            <CustomCard title="Hướng dẫn sử dụng">
              <Space direction="vertical">
                <div>
                  <Text strong>1. Đồng bộ toàn bộ:</Text>
                  <br />
                  <Text type="secondary">
                    Đồng bộ tất cả dữ liệu: repository, branches, commits, issues, pull requests
                  </Text>
                </div>
                
                <div>
                  <Text strong>2. Đồng bộ cơ bản:</Text>
                  <br />
                  <Text type="secondary">
                    Chỉ đồng bộ thông tin repository (nhanh, ít tốn API calls)
                  </Text>
                </div>
                
                <div>
                  <Text strong>3. Đồng bộ nâng cao:</Text>
                  <br />
                  <Text type="secondary">
                    Đồng bộ repository và branches với thông tin chi tiết
                  </Text>
                </div>
              </Space>
            </CustomCard>

            {/* Note */}
            <Alert
              type="info"
              message="Lưu ý"
              description="Đồng bộ toàn bộ có thể mất một thời gian tùy vào kích thước repository. GitHub API có giới hạn 5000 requests/hour cho authenticated users."
              showIcon
            />

          </Space>
        </Col>
      </Row>
    </div>
  );
};

export default SyncPage;