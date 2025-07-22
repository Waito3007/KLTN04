import React from 'react';
import { Card, Col, Row, Empty, Typography } from 'antd';
import ChartWrapper from './ChartWrapper';

const { Title } = Typography;

/**
 * Component hiển thị biểu đồ loại commit
 */
const CommitAnalyst = ({ memberCommits, allRepoCommitAnalysis }) => {
  let dataToDisplay = null;
  let cardTitle = "🏷️ Loại Commit";
  let showEmpty = false;

  if (memberCommits && memberCommits.statistics && memberCommits.statistics.commit_types && Object.keys(memberCommits.statistics.commit_types).length > 0) {
    dataToDisplay = memberCommits.statistics.commit_types;
    cardTitle = "🏷️ Loại Commit (Thành viên)";
  } else if (allRepoCommitAnalysis && allRepoCommitAnalysis.statistics && allRepoCommitAnalysis.statistics.commit_types && Object.keys(allRepoCommitAnalysis.statistics.commit_types).length > 0) {
    dataToDisplay = allRepoCommitAnalysis.statistics.commit_types;
    cardTitle = "🏷️ Loại Commit (Toàn bộ kho lưu trữ)";
  } else {
    showEmpty = true;
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