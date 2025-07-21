import { useState } from 'react';
import { Button, Badge, Popover, List, Typography, Divider, Spin, Tag, Alert, Tooltip, Modal } from 'antd';
import { ExclamationCircleFilled, CheckCircleFilled, InfoCircleOutlined, FileTextOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Text, Title } = Typography;

const AnalyzeGitHubCommits = ({ repo }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [popoverVisible, setPopoverVisible] = useState(false);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [currentDiff, setCurrentDiff] = useState('');

  const analyzeCommits = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = localStorage.getItem("access_token");
      
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await axios.get(
        `http://localhost:8000/api/commits/analyze-github/${repo.owner.login}/${repo.name}`,
        {
          headers: { 
            Authorization: `Bearer ${token}`,
            Accept: "application/json"
          },
          params: { 
            per_page: 10,
            include_diff: true, // Request full diff content
            // Add cache busting to avoid stale data
            timestamp: Date.now()
          },
          timeout: 10000 // 10 second timeout
        }
      );
      
      if (!response.data) {
        throw new Error('Invalid response data');
      }

      setAnalysis(response.data);
    } catch (err) {
      let errorMessage = 'Failed to analyze commits';
      
      if (err.response) {
        if (err.response.status === 401) {
          errorMessage = 'Please login to analyze commits';
        } else if (err.response.status === 403) {
          errorMessage = 'Access to this repository is denied';
        } else if (err.response.data?.detail) {
          errorMessage = err.response.data.detail;
        }
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handlePopoverOpen = (visible) => {
    setPopoverVisible(visible);
    if (visible && !analysis && !error) {
      analyzeCommits();
    }
  };

  const getStatusColor = () => {
    if (error) return 'warning';
    if (!analysis) return 'default';
    return analysis.critical > 0 ? 'error' : 'success';
  };

  const getStatusText = () => {
    if (error) return 'Error';
    if (!analysis) return 'Analyze Commits';
    return analysis.critical > 0 
      ? `${analysis.critical} Critical` 
      : 'No Issues';
  };

  const getStatusIcon = () => {
    if (error) return <InfoCircleOutlined />;
    if (!analysis) return null;
    return analysis.critical > 0 
      ? <ExclamationCircleFilled /> 
      : <CheckCircleFilled />;
  };
  const handleShowDiff = (diffContent) => {
    setCurrentDiff(diffContent);
    setIsModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsModalVisible(false);
    setCurrentDiff('');
  };

  const renderContent = () => {
    if (loading) {
      return (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin size="small" />
          <div style={{ marginTop: 8 }}>
            <Text type="secondary">Analyzing commits...</Text>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <Alert
          message="Analysis Failed"
          description={error}
          type="error"
          showIcon
        />
      );
    }

    if (!analysis) {
      return <Text type="secondary">Click to analyze commits</Text>;
    }

    return (
      <>
        <div style={{ marginBottom: 16 }}>
          <Title level={5} style={{ marginBottom: 4 }}>
            Commit Analysis Summary
          </Title>
          <Text>
            <Tag color={analysis.critical > 0 ? 'error' : 'success'}>
              {analysis.critical > 0 ? 'Needs Review' : 'All Clear'}
            </Tag>
            {analysis.critical} of {analysis.total} commits are critical
          </Text>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        <List
          size="small"
          dataSource={analysis.details.slice(0, 5)}
          renderItem={item => (
            <List.Item>
              <div style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Tag color={item.is_critical ? 'error' : 'success'}>
                    {item.is_critical ? 'CRITICAL' : 'Normal'}
                  </Tag>
                  <Tooltip title="Commit ID">
                    <Text code style={{ fontSize: 12 }}>
                      {item.id.substring(0, 7)}
                    </Text>
                  </Tooltip>
                </div>
                <Text
                  ellipsis={{ tooltip: item.message_preview }}
                  style={{ 
                    color: item.is_critical ? '#f5222d' : 'inherit',
                    marginTop: 4,
                    display: 'block'
                  }}
                >
                  {item.message_preview}
                </Text>
                {item.diff_content && item.diff_content !== "Error fetching diff" && (
                  <Button 
                    type="link" 
                    size="small" 
                    icon={<FileTextOutlined />} 
                    onClick={() => handleShowDiff(item.diff_content)}
                    style={{ paddingLeft: 0 }}
                  >
                    View Diff
                  </Button>
                )}
              </div>
            </List.Item>
          )}
        />

        {analysis.total > 5 && (
          <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
            Showing 5 of {analysis.total} commits
          </Text>
        )}
      </>
    );
  };

  return (
    <>
      <Popover 
        content={renderContent()}
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Commit Analysis</span>
            {analysis && (
              <Badge 
                count={`${analysis.critical_percentage}%`} 
                style={{ 
                  backgroundColor: analysis.critical > 0 ? '#f5222d' : '#52c41a'
                }} 
              />
            )}
          </div>
        }
        trigger="click"
        open={popoverVisible}
        onOpenChange={handlePopoverOpen}
        overlayStyle={{ width: 350 }}
        placement="bottomRight"
      >
        <Badge 
          count={analysis?.critical || 0} 
          color={getStatusColor()}
          offset={[-10, 10]}
        >
          <Button 
            type={error ? 'default' : analysis ? (analysis.critical ? 'danger' : 'success') : 'default'}
            icon={getStatusIcon()}
            loading={loading}
            onClick={(e) => e.stopPropagation()}
            style={{ 
              marginLeft: 'auto',
              fontWeight: 500,
              borderRadius: 20,
              padding: '0 16px',
              border: error ? '1px solid #faad14' : undefined
            }}
          >
            {getStatusText()}
          </Button>
        </Badge>
      </Popover>

      <Modal
        title="Commit Diff"
        open={isModalVisible}
        onCancel={handleCloseModal}
        footer={null}
        width={800}
        style={{ top: 20 }}
        bodyStyle={{ maxHeight: 'calc(100vh - 200px)', overflowY: 'auto' }}
      >
        <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          {currentDiff}
        </pre>
      </Modal>
    </>
  );
};

export default AnalyzeGitHubCommits;