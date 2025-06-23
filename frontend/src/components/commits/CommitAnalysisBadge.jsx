// components/CommitAnalysisBadge.jsx
import { Tag, Tooltip, Popover, List, Typography, Divider, Badge, Spin } from 'antd';
import { ExclamationCircleFilled, CheckCircleFilled } from '@ant-design/icons';
import { useState } from 'react';
import axios from 'axios';

const { Text } = Typography;

const CommitAnalysisBadge = ({ repo }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchCommitAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = localStorage.getItem("access_token");
      const response = await axios.get(
        `http://localhost:8000/api/github/repos/${repo.owner.login}/${repo.name}/commits`,
        {
          headers: { Authorization: `token ${token}` },
          params: { per_page: 5 } // Get last 5 commits for analysis
        }
      );
      
      // Analyze the commits
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

  const getStatusColor = () => {
    if (!analysis) return 'default';
    return analysis.critical > 0 ? 'error' : 'success';
  };

  const getStatusText = () => {
    if (!analysis) return 'Analyze Commits';
    return analysis.critical > 0 
      ? `${analysis.critical} Critical Commits` 
      : 'No Critical Commits';
  };

  const getStatusIcon = () => {
    if (!analysis) return null;
    return analysis.critical > 0 
      ? <ExclamationCircleFilled /> 
      : <CheckCircleFilled />;
  };

  const content = (
    <div style={{ maxWidth: 300 }}>
      {loading && <Spin size="small" />}
      {error && <Text type="danger">{error}</Text>}
      {analysis && (
        <>
          <Text strong>Recent Commits Analysis</Text>
          <Divider style={{ margin: '8px 0' }} />
          <List
            size="small"
            dataSource={analysis.details.slice(0, 5)}
            renderItem={item => (
              <List.Item>
                <div style={{ width: '100%' }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    marginBottom: 4
                  }}>
                    <Text 
                      ellipsis 
                      style={{ 
                        maxWidth: 180,
                        color: item.is_critical ? '#f5222d' : 'inherit'
                      }}
                    >
                      {item.message_preview}
                    </Text>
                    <Tag color={item.is_critical ? 'error' : 'success'}>
                      {item.is_critical ? 'Critical' : 'Normal'}
                    </Tag>
                  </div>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {item.id.substring(0, 7)}
                  </Text>
                </div>
              </List.Item>
            )}
          />
          <Divider style={{ margin: '8px 0' }} />
          <Text type="secondary">
            {analysis.critical} of {analysis.total} recent commits are critical
          </Text>
        </>
      )}
    </div>
  );

  return (
    <Popover 
      content={content}
      title="Commit Analysis"
      trigger="click"
      onVisibleChange={visible => visible && !analysis && fetchCommitAnalysis()}
    >
      <Badge 
        count={analysis?.critical || 0} 
        style={{ backgroundColor: getStatusColor() }}
      >
        <Tag 
          icon={getStatusIcon()}
          color={getStatusColor()}
          style={{ cursor: 'pointer' }}
        >
          {getStatusText()}
        </Tag>
      </Badge>
    </Popover>
  );
};

export default CommitAnalysisBadge;