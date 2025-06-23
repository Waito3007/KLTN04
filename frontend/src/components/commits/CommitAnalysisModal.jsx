// components/CommitAnalysisModal.jsx
import { Modal, List, Typography, Tag, Divider, Spin, Tabs, Progress, Alert } from 'antd';
import { 
  ExclamationCircleOutlined, 
  CheckCircleOutlined,
  BarChartOutlined,
  FileTextOutlined 
} from '@ant-design/icons';
import axios from 'axios';
import { useState, useEffect } from 'react';

const { Title, Text } = Typography;

const CommitAnalysisModal = ({ repo, visible, onCancel }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchFullAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = localStorage.getItem("access_token");
      const response = await axios.get(
        `http://localhost:8000/api/github/repos/${repo.owner.login}/${repo.name}/commits`,
        {
          headers: { Authorization: `token ${token}` },
          params: { per_page: 100 } // Get more commits for detailed analysis
        }
      );
      
      const analysisRes = await axios.post(
        'http://localhost:8000/api/commits/analyze-json',
        {
          commits: response.data.map(commit => ({
            id: commit.sha,
            message: commit.commit.message
          }))
        },
        {
          headers: { Authorization: `token ${token}` }
        }
      );
      
      setAnalysis(analysisRes.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze commits');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (visible) {
      fetchFullAnalysis();
    }
  }, [visible]);

  const criticalPercentage = analysis 
    ? Math.round((analysis.critical / analysis.total) * 100) 
    : 0;

  return (
    <Modal
      title={<><BarChartOutlined /> Commit Analysis for {repo.name}</>}
      visible={visible}
      onCancel={onCancel}
      footer={null}
      width={800}
    >
      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}
      
      {error && (
        <Alert 
          message="Error" 
          description={error} 
          type="error" 
          showIcon 
          style={{ marginBottom: 20 }}
        />
      )}
        {analysis && (
        <Tabs 
          defaultActiveKey="1"
          items={[
            {
              key: '1',
              label: (
                <span>
                  <FileTextOutlined /> Commits
                </span>
              ),
              children: (
                <div style={{ marginBottom: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                    <Progress
                      type="circle"
                      percent={criticalPercentage}
                      width={80}
                      format={percent => (
                        <Text strong style={{ fontSize: 24, color: percent > 0 ? '#f5222d' : '#52c41a' }}>
                          {percent}%
                        </Text>
                      )}
                      status={criticalPercentage > 0 ? 'exception' : 'success'}
                    />
                    <div style={{ marginLeft: 20 }}>
                      <Title level={4} style={{ marginBottom: 0 }}>
                        {analysis.critical} of {analysis.total} commits are critical
                      </Title>
                      <Text type="secondary">
                        {criticalPercentage > 0 
                          ? 'This repository contains potentially critical changes'
                          : 'No critical commits detected'}
                      </Text>
                    </div>
                  </div>
                  
                  <List
                    size="large"
                    dataSource={analysis.details}
                    renderItem={item => (
                      <List.Item>
                        <div style={{ width: '100%' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Tag color={item.is_critical ? 'error' : 'success'}>
                              {item.is_critical ? 'CRITICAL' : 'Normal'}
                            </Tag>
                            <Text type="secondary" copyable>
                              {item.id.substring(0, 7)}
                            </Text>
                          </div>
                          <Divider style={{ margin: '8px 0' }} />
                          <Text style={{ color: item.is_critical ? '#f5222d' : 'inherit' }}>
                            {item.message_preview}
                          </Text>
                        </div>
                      </List.Item>
                    )}
                  />
                </div>
              )
            },
            {
              key: '2', 
              label: (
                <span>
                  <ExclamationCircleOutlined /> Critical Commits
                </span>
              ),
              children: analysis.critical > 0 ? (
                <List
                  dataSource={analysis.details.filter(c => c.is_critical)}
                  renderItem={item => (
                    <List.Item>
                      <Alert
                        message="Critical Commit"
                        description={
                          <>
                            <Text strong style={{ display: 'block', marginBottom: 4 }}>
                              {item.message_preview}
                            </Text>
                            <Text type="secondary">Commit ID: {item.id.substring(0, 7)}</Text>
                          </>
                        }
                        type="error"
                        showIcon
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px 0' }}>
                  <CheckCircleOutlined style={{ fontSize: 48, color: '#52c41a', marginBottom: 20 }} />
                  <Title level={4} style={{ color: '#52c41a' }}>
                    No Critical Commits Found
                  </Title>
                  <Text type="secondary">
                    All analyzed commits appear to be normal changes
                  </Text>
                </div>
              )
            }
          ]}
        />
      )}
    </Modal>
  );
};

export default CommitAnalysisModal;