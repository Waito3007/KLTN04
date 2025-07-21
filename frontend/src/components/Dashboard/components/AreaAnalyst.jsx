import React from 'react';
import { Card, Col, Row } from 'antd';
import ChartWrapper from './ChartWrapper';

/**
 * Component hiển thị biểu đồ lĩnh vực công nghệ
 */
const AreaAnalyst = ({ memberCommits }) => {
  if (!memberCommits || !memberCommits.statistics || !memberCommits.statistics.tech_analysis) return null;

  const techChartData = {
    labels: Object.keys(memberCommits.statistics.tech_analysis),
    datasets: [{
      label: 'Số lượng',
      data: Object.values(memberCommits.statistics.tech_analysis),
      backgroundColor: '#1890ff',
      borderColor: '#0056b3',
      borderWidth: 2,
    }]
  };

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Card title="🌐 Lĩnh vực công nghệ" size="small">
          <ChartWrapper
            type="bar"
            data={techChartData}
            options={{
              plugins: {
                legend: {
                  position: 'top',
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

export default AreaAnalyst;
