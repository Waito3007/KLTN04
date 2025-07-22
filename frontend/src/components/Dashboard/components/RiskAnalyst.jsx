import React from 'react';
import { Card, Col, Row, Typography, Empty, Spin } from 'antd';
import ChartWrapper from './ChartWrapper';

const { Title } = Typography;

/**
 * Component hiển thị biểu đồ phân tích rủi ro
 */
const RiskAnalyst = ({ memberCommits, fullRiskAnalysis, fullAnalysisLoading }) => {

  let dataToDisplay = null;
  let cardTitle = "⚠️ Phân tích rủi ro";
  let showEmpty = false;

  if (fullAnalysisLoading) {
    return (
      <Card title={cardTitle} size="small">
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin tip="Đang tải phân tích rủi ro toàn bộ kho lưu trữ..." />
        </div>
      </Card>
    );
  }

  if (memberCommits && memberCommits.statistics && memberCommits.statistics.risk_analysis) {
    dataToDisplay = memberCommits.statistics.risk_analysis;
    cardTitle = "⚠️ Phân tích rủi ro (Thành viên)";
  } else if (fullRiskAnalysis && fullRiskAnalysis.risk_distribution) {
    dataToDisplay = fullRiskAnalysis.risk_distribution;
    cardTitle = "⚠️ Phân tích rủi ro (Toàn bộ kho lưu trữ)";
  } else {
    showEmpty = true;
  }

  if (showEmpty || !dataToDisplay || Object.keys(dataToDisplay).length === 0) {
    return (
      <Card title={cardTitle} size="small">
        <Empty description="Không có dữ liệu phân tích rủi ro" />
      </Card>
    );
  }

  const riskChartData = {
    labels: Object.keys(dataToDisplay),
    datasets: [{
      label: 'Số lượng',
      data: Object.values(dataToDisplay),
      backgroundColor: ['#52c41a', '#ff4d4f', '#d9d9d9'], // Green for lowrisk, Red for highrisk, Grey for unknown
      borderColor: ['#389e08', '#cf1322', '#bfbfbf'],
      borderWidth: 2,
    }]
  };

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Card title={cardTitle} size="small">
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
