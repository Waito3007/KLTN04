import React from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { ProjectOutlined, CheckCircleOutlined, WarningOutlined } from '@ant-design/icons';

const OverviewCard = ({ projects = 10, completedTasks = 50, overdueTasks = 5 }) => {
  return (
    <Card title="Tổng quan dự án" bordered={false}>
      <Row gutter={16}>
        <Col span={8}>
          <Statistic
            title="Số dự án"
            value={projects}
            prefix={<ProjectOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Công việc hoàn thành"
            value={completedTasks}
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Công việc trễ hạn"
            value={overdueTasks}
            prefix={<WarningOutlined />}
            valueStyle={{ color: '#ff4d4f' }}
          />
        </Col>
      </Row>
    </Card>
  );
};

export default OverviewCard;