import React from 'react';
import { Card, Space, Typography, Button, Tag } from 'antd';
import { BulbOutlined, WarningOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Title, Text } = Typography;

// Styled components
const InsightContainer = styled(Card)`
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  background: #ffffff;
  transition: all 0.3s ease;

  &:hover {
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    transform: translateY(-2px);
  }
`;

const InsightCard = styled(Card)`
  border-radius: 8px;
  border: 1px solid ${(props) => props.borderColor || '#f0f0f0'};
  background: #fff;
  transition: all 0.3s ease;
  padding: 12px;

  &:hover {
    border-color: ${(props) => props.borderColor || '#d9d9d9'};
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  @media (max-width: 576px) {
    padding: 8px;
  }
`;

const IconWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: ${(props) => props.bgColor || '#f0f0f0'};
`;

const ActionWrapper = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 8px;

  @media (max-width: 576px) {
    justify-content: flex-start;
    margin-top: 8px;
  }
`;

const AIInsightWidget = () => {
  const insights = [
    {
      id: 1,
      type: 'suggestion',
      title: 'Phân công đề xuất',
      description: 'Thêm 2 developer vào repo "frontend" để đảm bảo deadline 25/04/2025.',
    },
    {
      id: 2,
      type: 'warning',
      title: 'Dự đoán tiến độ',
      description: 'Repo "backend" có nguy cơ trễ hạn 3 ngày. Xem xét tăng tài nguyên.',
    },
  ];

  const getInsightStyle = (type) => {
    switch (type) {
      case 'suggestion':
        return {
          icon: <BulbOutlined style={{ fontSize: 20, color: '#1890ff' }} />,
          tag: <Tag color="blue">Đề xuất</Tag>,
          borderColor: '#e6f7ff',
          iconBg: '#e6f7ff',
        };
      case 'warning':
        return {
          icon: <WarningOutlined style={{ fontSize: 20, color: '#fa8c16' }} />,
          tag: <Tag color="orange">Cảnh báo</Tag>,
          borderColor: '#fff7e6',
          iconBg: '#fff7e6',
        };
      default:
        return {
          icon: null,
          tag: null,
          borderColor: '#f0f0f0',
          iconBg: '#f0f0f0',
        };
    }
  };

  return (
    <InsightContainer
      title={<Title level={4} style={{ margin: 0 }}>Gợi ý AI</Title>}
      bordered={false}
    >
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        {insights.map((item) => {
          const { icon, tag, borderColor, iconBg } = getInsightStyle(item.type);
          return (
            <InsightCard key={item.id} borderColor={borderColor}>
              <Space direction="horizontal" size="middle" style={{ width: '100%', alignItems: 'center' }}>
                <IconWrapper bgColor={iconBg}>{icon}</IconWrapper>
                <Space direction="vertical" size={4} style={{ flex: 1 }}>
                  <Space>
                    <Title level={5} style={{ margin: 0 }}>{item.title}</Title>
                    {tag}
                  </Space>
                  <Text type="secondary">{item.description}</Text>
                </Space>
                <ActionWrapper>
                  <Button type="primary" size="small">Thực hiện</Button>
                  <Button size="small">Bỏ qua</Button>
                </ActionWrapper>
              </Space>
            </InsightCard>
          );
        })}
      </Space>
    </InsightContainer>
  );
};

export default AIInsightWidget;