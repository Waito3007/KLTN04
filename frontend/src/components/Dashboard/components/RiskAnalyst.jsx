import React from 'react';
import { Card, Col, Row } from 'antd';
import ChartWrapper from './ChartWrapper';

/**
 * Component hiển thị biểu đồ phân tích rủi ro
 */
const RiskAnalyst = ({ memberCommits }) => {
  if (!memberCommits || !memberCommits.statistics || !memberCommits.statistics.risk_analysis) return null;

  const riskChartData = {
    labels: Object.keys(memberCommits.statistics.risk_analysis),
    datasets: [{
      label: 'Số lượng',
      data: Object.values(memberCommits.statistics.risk_analysis),
      backgroundColor: ['#52c41a', '#ff4d4f'], // Green for lowrisk, Red for highrisk
      borderColor: ['#389e08', '#cf1322'],
      borderWidth: 2,
    }]
  };

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Card title="⚠️ Phân tích rủi ro" size="small">
          <ChartWrapper
            type="pie"
            data={riskChartData}
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

export default RiskAnalyst;
