import React from 'react';
import { Card, Statistic, Space, Row, Col } from 'antd';
import { ProjectOutlined, CheckCircleOutlined, WarningOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const SidebarOverviewCard = styled(Card)`
  .ant-card-body {
    padding: 16px;
  }
`;

const StatisticItem = styled.div`
  padding: 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  margin-bottom: 12px;
  transition: all 0.2s ease;

  &:hover {
    border-color: #3b82f6;
    transform: translateY(-1px);
  }

  &:last-child {
    margin-bottom: 0;
  }
`;

const OverviewCard = ({ projects = 10, completedTasks = 50, overdueTasks = 5, sidebar = false }) => {
  if (sidebar) {
    return (
      <SidebarOverviewCard 
        title="Tổng quan dự án" 
        variant="outlined"
        size="small"
      >
        <Space direction="vertical" style={{ width: '100%' }} size={0}>
          <StatisticItem>
            <Statistic
              title="Số dự án"
              value={projects}
              prefix={<ProjectOutlined />}
              valueStyle={{ color: '#1890ff', fontSize: '18px' }}
            />
          </StatisticItem>
          <StatisticItem>
            <Statistic
              title="Công việc hoàn thành"
              value={completedTasks}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a', fontSize: '18px' }}
            />
          </StatisticItem>
          <StatisticItem>
            <Statistic
              title="Công việc trễ hạn"
              value={overdueTasks}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#ff4d4f', fontSize: '18px' }}
            />
          </StatisticItem>
        </Space>
      </SidebarOverviewCard>
    );
  }  // Layout ngang cho desktop thường
  return (
    <Card title="Tổng quan dự án" variant="outlined">
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