import React from 'react';
import { Card, Col, Row, Empty, Typography } from 'antd';
import ChartWrapper from './ChartWrapper';

const { Title } = Typography;

/**
 * Component hiển thị biểu đồ loại commit
 */
const CommitAnalyst = ({ memberCommits, allRepoCommitAnalysis, aiModel, loading }) => {

  let dataToDisplay = null;
  let cardTitle = "🏷️ Loại Commit";
  let showEmpty = false;
  // Use loading prop directly from arguments

  // Logic hiển thị theo model và tương thích kiểu dữ liệu backend
  function normalizeCommitTypes(commitTypes) {
    if (Array.isArray(commitTypes)) {
      // Nếu backend trả về dạng array: [{type: 'feat', count: 10}, ...]
      const obj = {};
      commitTypes.forEach(item => {
        if (item.type && typeof item.count === 'number') {
          obj[item.type] = item.count;
        }
      });
      return obj;
    }
    // Nếu là object thì trả về luôn
    return commitTypes;
  }

  // Hiển thị loading khi đang tải dữ liệu
  if (loading) {
    return (
      <Card title={cardTitle} size="small">
        <div style={{ textAlign: 'center', padding: '32px 0' }}>
          <span>Đang tải dữ liệu phân tích commit...</span>
        </div>
      </Card>
    );
  }
  if (aiModel === 'multifusion') {
    if (allRepoCommitAnalysis && allRepoCommitAnalysis.statistics && allRepoCommitAnalysis.statistics.commit_types) {
      const normalized = normalizeCommitTypes(allRepoCommitAnalysis.statistics.commit_types);
      if (normalized && Object.keys(normalized).length > 0) {
        dataToDisplay = normalized;
        cardTitle = "🏷️ Loại Commit (MultiFusion - Toàn bộ kho lưu trữ)";
      } else {
        showEmpty = true;
      }
    } else {
      showEmpty = true;
    }
  } else {
    if (memberCommits && memberCommits.statistics && memberCommits.statistics.commit_types) {
      const normalized = normalizeCommitTypes(memberCommits.statistics.commit_types);
      if (normalized && Object.keys(normalized).length > 0) {
        dataToDisplay = normalized;
        cardTitle = "🏷️ Loại Commit (HAN - Thành viên)";
      } else {
        showEmpty = true;
      }
    } else {
      showEmpty = true;
    }
  }

  if (showEmpty || !dataToDisplay || Object.keys(dataToDisplay).length === 0) {
    return (
      <Card title={cardTitle} size="small">
        <Empty description="Không có dữ liệu phân tích loại commit." />
      </Card>
    );
  }

  const typeChartData = {
    labels: Object.keys(dataToDisplay),
    datasets: [{
      data: Object.values(dataToDisplay),
      backgroundColor: [
        '#52c41a', '#f5222d', '#1890ff', '#fa8c16', 
        '#722ed1', '#13c2c2', '#eb2f96', '#666666',
        '#fadb14', '#a0d911', '#ff7a45', '#ff85c0'
      ]
    }]
  };

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Card title={cardTitle} size="small">
          <ChartWrapper
            type="pie"
            data={typeChartData}
            options={{
              plugins: {
                legend: {
                  position: 'bottom',
                  labels: {
                    color: '#333',
                    font: {
                      size: 14,
                      weight: 'bold',
                    }
                  }
                }
              }
            }}
            style={{ height: '300px', display: 'flex', justifyContent: 'center' }}
          />
        </Card>
      </Col>
    </Row>
  );
};

export default CommitAnalyst;