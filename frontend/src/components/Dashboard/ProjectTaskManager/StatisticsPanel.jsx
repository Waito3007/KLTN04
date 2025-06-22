import React from 'react';
import { Card, Row, Col, Statistic, Progress } from 'antd';
import { BarChartOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';

const StatisticsPanel = ({ stats = {} }) => {
  // Fallback values nếu stats undefined
  const safeStats = {
    total: stats?.total || 0,
    completed: stats?.completed || 0,
    inProgress: stats?.inProgress || 0,
    todo: stats?.todo || 0,
    completionPercentage: stats?.completionPercentage || 0
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <Row gutter={16}>
        <Col span={6}>
          <Card size="small">
            <Statistic 
              title="Tổng tasks" 
              value={safeStats.total}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic 
              title="Hoàn thành" 
              value={safeStats.completed}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>        <Col span={6}>
          <Card size="small">
            <Statistic 
              title="Đang làm" 
              value={safeStats.inProgress}
              valueStyle={{ color: '#1890ff' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic 
              title="Tỷ lệ hoàn thành" 
              value={safeStats.completionPercentage}
              suffix="%"
              valueStyle={{ color: safeStats.completionPercentage > 70 ? '#52c41a' : '#fa8c16' }}
            />
            <Progress 
              percent={safeStats.completionPercentage} 
              showInfo={false}
              size="small"
              strokeColor={safeStats.completionPercentage > 70 ? '#52c41a' : '#fa8c16'}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default StatisticsPanel;
