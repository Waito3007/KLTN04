import React from 'react';
import { Card, Col, Row, Typography, Empty, Spin } from 'antd';
import ChartWrapper from './ChartWrapper';

const { Title } = Typography;

/**
 * Component hiển thị biểu đồ lĩnh vực công nghệ
 */
const AreaAnalyst = ({ memberCommits, fullAreaAnalysis, fullAnalysisLoading }) => {

  let dataToDisplay = null;
  let cardTitle = "🌐 Lĩnh vực công nghệ";
  let showEmpty = false;

  if (fullAnalysisLoading) {
    return (
      <Card title={cardTitle} size="small">
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin tip="Đang tải phân tích lĩnh vực toàn bộ kho lưu trữ..." />
        </div>
      </Card>
    );
  }

  if (memberCommits && memberCommits.statistics && memberCommits.statistics.tech_analysis) {
    dataToDisplay = memberCommits.statistics.tech_analysis;
    cardTitle = "🌐 Lĩnh vực công nghệ (Thành viên)";
  } else if (fullAreaAnalysis && fullAreaAnalysis.area_distribution) {
    dataToDisplay = fullAreaAnalysis.area_distribution;
    cardTitle = "🌐 Lĩnh vực công nghệ (Toàn bộ kho lưu trữ)";
  } else {
    showEmpty = true;
  }

  if (showEmpty || !dataToDisplay || Object.keys(dataToDisplay).length === 0) {
    return (
      <Card title={cardTitle} size="small">
        <Empty description="Không có dữ liệu phân tích lĩnh vực" />
      </Card>
    );
  }

  const techChartData = {
    labels: Object.keys(dataToDisplay),
    datasets: [{
      label: 'Số lượng',
      data: Object.values(dataToDisplay),
      backgroundColor: '#1890ff',
      borderColor: '#0056b3',
      borderWidth: 2,
    }]
  };

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Card title={cardTitle} size="small">
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
