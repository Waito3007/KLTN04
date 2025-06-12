import React from 'react';
import { Card, Row, Col, Statistic, Progress } from 'antd';
import { BarChartOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';

const StatisticsPanel = ({ stats }) => (
  <div style={{ marginBottom: 16 }}>
    <Row gutter={16}>
      <Col span={6}>
        <Card size="small">
          <Statistic 
            title="Tổng tasks" 
            value={stats.total}
            prefix={<BarChartOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small">
          <Statistic 
            title="Hoàn thành" 
            value={stats.completed}
            valueStyle={{ color: '#52c41a' }}
            prefix={<CheckCircleOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small">
          <Statistic 
            title="Đang làm" 
            value={stats.inProgress}
            valueStyle={{ color: '#1890ff' }}
            prefix={<ExclamationCircleOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card size="small">
          <Statistic 
            title="Tỷ lệ hoàn thành" 
            value={stats.completionRate}
            suffix="%"
            valueStyle={{ color: stats.completionRate > 70 ? '#52c41a' : '#fa8c16' }}
          />
          <Progress 
            percent={stats.completionRate} 
            showInfo={false}
            size="small"
            strokeColor={stats.completionRate > 70 ? '#52c41a' : '#fa8c16'}
          />
        </Card>
      </Col>
    </Row>
  </div>
);

export default StatisticsPanel;
