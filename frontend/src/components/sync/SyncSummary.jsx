import React from 'react';
import { Row, Col, Statistic } from 'antd';
import { GithubOutlined, ExclamationCircleOutlined, ClockCircleOutlined, CheckCircleOutlined } from '@ant-design/icons';
import styled from 'styled-components';
import { Card } from '@components/common';

const StatsCard = styled(Card)`
  text-align: center;
  border-radius: 8px;
  .ant-statistic-content-value {
    color: ${props => props.color || '#1890ff'};
  }
`;

const SyncSummary = ({ summary }) => {
  if (!summary) return null;

  return (
    <Row gutter={16} style={{ marginBottom: 24 }}>
      <Col span={6}>
        <StatsCard color="#1890ff">
          <Statistic 
            title="Tổng GitHub Repos" 
            value={summary.total_github_repos}
            prefix={<GithubOutlined />}
          />
        </StatsCard>
      </Col>
      <Col span={6}>
        <StatsCard color="#ff4d4f">
          <Statistic 
            title="Chưa đồng bộ" 
            value={summary.unsynced_count}
            prefix={<ExclamationCircleOutlined />}
          />
        </StatsCard>
      </Col>
      <Col span={6}>
        <StatsCard color="#faad14">
          <Statistic 
            title="Cần cập nhật" 
            value={summary.outdated_count}
            prefix={<ClockCircleOutlined />}
          />
        </StatsCard>
      </Col>
      <Col span={6}>
        <StatsCard color="#52c41a">
          <Statistic 
            title="Đã đồng bộ" 
            value={summary.synced_count}
            prefix={<CheckCircleOutlined />}
          />
        </StatsCard>
      </Col>
    </Row>
  );
};

export default SyncSummary;
