// frontend/src/components/RepositorySync.jsx
import React, { useState } from 'react';
import { Button, Card, Progress, Alert, Descriptions, Space, Typography } from 'antd';
import { SyncOutlined, DatabaseOutlined, GithubOutlined, BranchesOutlined, CodeOutlined, IssuesCloseOutlined, PullRequestOutlined } from '@ant-design/icons';
import { syncAPI } from "@services/api";
import { Loading, Toast } from '@components/common';

const { Title, Text } = Typography;

const RepositorySync = ({ owner, repoName, onSyncComplete }) => {
  const [loading, setLoading] = useState(false);
  const [syncResults, setSyncResults] = useState(null);
  const [syncType, setSyncType] = useState(null);
  const [error, setError] = useState(null);

  const handleSyncAll = async () => {
    setLoading(true);
    setError(null);
    setSyncResults(null);
    setSyncType('complete');

    try {
      Toast.info('Bắt đầu đồng bộ toàn bộ repository...');
      const result = await syncAPI.syncAll(owner, repoName);
      
      setSyncResults(result);
      Toast.success('Đồng bộ toàn bộ hoàn tất!');
      
      if (onSyncComplete) {
        onSyncComplete(result);
      }
    } catch (error) {
      console.error('Sync error:', error);
      setError(error.response?.data?.detail || error.message);
      Toast.error('Có lỗi xảy ra khi đồng bộ');
    } finally {
      setLoading(false);
    }
  };

  const handleSyncBasic = async () => {
    setLoading(true);
    setError(null);
    setSyncResults(null);
    setSyncType('basic');

    try {
      Toast.info('Bắt đầu đồng bộ cơ bản...');
      const result = await syncAPI.syncBasic(owner, repoName);
      
      setSyncResults(result);
      Toast.success('Đồng bộ cơ bản hoàn tất!');
      
      if (onSyncComplete) {
        onSyncComplete(result);
      }
    } catch (error) {
      console.error('Basic sync error:', error);
      setError(error.response?.data?.detail || error.message);
      Toast.error('Có lỗi xảy ra khi đồng bộ cơ bản');
    } finally {
      setLoading(false);
    }
  };

  const handleSyncEnhanced = async () => {
    setLoading(true);
    setError(null);
    setSyncResults(null);
    setSyncType('enhanced');

    try {
      Toast.info('Bắt đầu đồng bộ nâng cao...');
      const result = await syncAPI.syncEnhanced(owner, repoName);
      
      setSyncResults(result);
      Toast.success('Đồng bộ nâng cao hoàn tất!');
      
      if (onSyncComplete) {
        onSyncComplete(result);
      }
    } catch (error) {
      console.error('Enhanced sync error:', error);
      setError(error.response?.data?.detail || error.message);
      Toast.error('Có lỗi xảy ra khi đồng bộ nâng cao');
    } finally {
      setLoading(false);
    }
  };

  const renderSyncResults = () => {
    if (!syncResults) return null;

    const { sync_results } = syncResults;
    
    if (syncType === 'complete' && sync_results) {
      return (
        <Card 
          title={
            <Space>
              <DatabaseOutlined />
              <span>Kết quả đồng bộ toàn bộ</span>
            </Space>
          }
          style={{ marginTop: 16 }}
        >
          <Descriptions column={2} size="small">
            <Descriptions.Item 
              label={<Space><GithubOutlined />Repository</Space>}
              span={2}
            >
              {sync_results.repository_synced ? (
                <Text type="success">✅ Đã đồng bộ</Text>
              ) : (
                <Text type="danger">❌ Thất bại</Text>
              )}
            </Descriptions.Item>
            
            <Descriptions.Item 
              label={<Space><BranchesOutlined />Branches</Space>}
            >
              <Text type="success">{sync_results.branches_synced || 0} branches</Text>
            </Descriptions.Item>
            
            <Descriptions.Item 
              label={<Space><CodeOutlined />Commits với Diff</Space>}
            >
              <Text type="success">{sync_results.commits_synced || 0} commits</Text>
              {sync_results.commits_synced > 0 && (
                <div style={{ fontSize: '12px', color: '#666' }}>
                  Bao gồm code diff và file changes
                </div>
              )}
            </Descriptions.Item>
            
            <Descriptions.Item 
              label={<Space><IssuesCloseOutlined />Issues</Space>}
            >
              <Text type="success">{sync_results.issues_synced || 0} issues</Text>
            </Descriptions.Item>
            
            <Descriptions.Item 
              label={<Space><PullRequestOutlined />Pull Requests</Space>}
            >
              <Text type="success">{sync_results.pull_requests_synced || 0} PRs</Text>
            </Descriptions.Item>
          </Descriptions>

          {sync_results.errors && sync_results.errors.length > 0 && (
            <Alert
              type="warning"
              message="Một số lỗi đã xảy ra"
              description={
                <ul>
                  {sync_results.errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              }
              style={{ marginTop: 16 }}
            />
          )}
        </Card>
      );
    }

    // Basic and Enhanced sync results
    return (
      <Card 
        title={
          <Space>
            <DatabaseOutlined />
            <span>Kết quả đồng bộ {syncType === 'basic' ? 'cơ bản' : 'nâng cao'}</span>
          </Space>
        }
        style={{ marginTop: 16 }}
      >
        <Alert
          type="success"
          message={syncResults.message}
          description={`Repository ${owner}/${repoName} đã được đồng bộ thành công`}
        />
      </Card>
    );
  };

  return (
    <Card 
      title={
        <Space>
          <SyncOutlined spin={loading} />
          <span>Đồng bộ Repository</span>
        </Space>
      }
    >
      <div style={{ marginBottom: 16 }}>
        <Title level={5}>Repository: {owner}/{repoName}</Title>
        <Text type="secondary">
          Chọn loại đồng bộ phù hợp với nhu cầu của bạn
        </Text>
      </div>

      <Space direction="vertical" style={{ width: '100%' }}>
        <Card size="small" style={{ background: '#f8f9fa' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>🚀 Đồng bộ toàn bộ</Text>
              <br />
              <Text type="secondary">
                Đồng bộ repository, branches, commits (với code diff), issues, và pull requests
              </Text>
              <br />
              <Text type="warning" style={{ fontSize: '12px' }}>
                ⚠️ Sẽ mất nhiều thời gian hơn do lấy code diff chi tiết
              </Text>
            </div>
            <Button 
              type="primary"
              icon={<DatabaseOutlined />}
              onClick={handleSyncAll}
              loading={loading && syncType === 'complete'}
              disabled={loading}
              block
            >
              Đồng bộ toàn bộ
            </Button>
          </Space>
        </Card>

        <Card size="small" style={{ background: '#f8f9fa' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>⚡ Đồng bộ cơ bản</Text>
              <br />
              <Text type="secondary">
                Chỉ đồng bộ thông tin repository (nhanh)
              </Text>
            </div>
            <Button 
              icon={<GithubOutlined />}
              onClick={handleSyncBasic}
              loading={loading && syncType === 'basic'}
              disabled={loading}
              block
            >
              Đồng bộ cơ bản
            </Button>
          </Space>
        </Card>

        <Card size="small" style={{ background: '#f8f9fa' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>🔥 Đồng bộ nâng cao</Text>
              <br />
              <Text type="secondary">
                Đồng bộ repository và branches với thông tin chi tiết
              </Text>
            </div>
            <Button 
              type="dashed"
              icon={<BranchesOutlined />}
              onClick={handleSyncEnhanced}
              loading={loading && syncType === 'enhanced'}
              disabled={loading}
              block
            >
              Đồng bộ nâng cao
            </Button>
          </Space>
        </Card>
      </Space>

      {loading && (
        <div style={{ textAlign: 'center', marginTop: 16 }}>
          <Loading variant="circle" size="large" message="Đang đồng bộ dữ liệu, vui lòng đợi..." />
        </div>
      )}

      {error && (
        <Alert
          type="error"
          message="Lỗi đồng bộ"
          description={error}
          style={{ marginTop: 16 }}
          closable
          onClose={() => setError(null)}
        />
      )}

      {renderSyncResults()}
    </Card>
  );
};

export default RepositorySync;
