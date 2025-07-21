import React from 'react';
import { Card, Col, Row } from 'antd';
import ChartWrapper from './ChartWrapper';

/**
 * Component hiển thị biểu đồ loại commit
 */
const CommitAnalyst = ({ memberCommits }) => {
  if (!memberCommits || !memberCommits.statistics || !memberCommits.statistics.commit_types) return null;

  const typeChartData = {
    labels: Object.keys(memberCommits.statistics.commit_types),
    datasets: [{
      data: Object.values(memberCommits.statistics.commit_types),
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
        <Card title="🏷️ Loại Commit" size="small">
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
