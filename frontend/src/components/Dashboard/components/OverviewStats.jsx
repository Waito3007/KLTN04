import React from 'react';
import { Row, Col, Card, Typography } from 'antd';

const { Title, Text } = Typography;

/**
 * Component hiển thị thống kê tổng quan về thành viên
 */
const OverviewStats = ({ members, branches }) => {
  if (!members || members.length === 0) return null;
  
  return (
    <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
      <Col xs={24} sm={8}>
        <Card size="small" style={{ textAlign: 'center' }}>
          <Title level={3} style={{ color: '#1890ff', margin: 0 }}>
            {members.length}
          </Title>
          <Text type="secondary">Thành viên tham gia</Text>
        </Card>
      </Col>
      <Col xs={24} sm={8}>
        <Card size="small" style={{ textAlign: 'center' }}>
          <Title level={3} style={{ color: '#52c41a', margin: 0 }}>
            {branches.length}
          </Title>
          <Text type="secondary">Nhánh trong dự án</Text>
        </Card>
      </Col>
      <Col xs={24} sm={8}>
        <Card size="small" style={{ textAlign: 'center' }}>
          <Title level={3} style={{ color: '#fa8c16', margin: 0 }}>
            {members.reduce((sum, member) => sum + (member.total_commits || 0), 0)}
          </Title>
          <Text type="secondary">Tổng commits</Text>
        </Card>
      </Col>
    </Row>
  );
};

export default OverviewStats;
