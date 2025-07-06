import React from 'react';
import { Card, Typography, Row, Col, Divider, Tag } from 'antd';

const { Title, Text } = Typography;

/**
 * Component hiển thị thông tin phân tích từ MultiFusion V2
 */
const MultiFusionInsights = ({ multifusionInsights }) => {
  if (!multifusionInsights) return null;

  return (
    <Card title="🔬 Phân tích nâng cao MultiFusion V2" style={{ marginTop: '20px' }}>
      <Row gutter={[16, 16]}>
        {/* Dominant Commit Type */}
        <Col span={8}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Title level={4} style={{ color: '#1890ff', margin: 0 }}>
              {multifusionInsights.dominant_commit_type?.type || 'N/A'}
            </Title>
            <Text type="secondary">Loại commit chủ đạo</Text>
            <br />
            <Text strong>
              {multifusionInsights.dominant_commit_type?.percentage || 0}%
            </Text>
            <Text type="secondary"> trong tổng số commits</Text>
          </Card>
        </Col>
        
        {/* Productivity Metrics */}
        <Col span={8}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Title level={4} style={{ color: '#52c41a', margin: 0 }}>
              {multifusionInsights.productivity_metrics?.avg_changes_per_commit?.toFixed(1) || 0}
            </Title>
            <Text type="secondary">Trung bình thay đổi/Commit</Text>
            <br />
            <Text strong>
              {multifusionInsights.productivity_metrics?.avg_files_per_commit?.toFixed(1) || 0}
            </Text>
            <Text type="secondary"> files/commit</Text>
          </Card>
        </Col>
        
        {/* Languages Used */}
        <Col span={8}>
          <Card size="small" style={{ textAlign: 'center' }}>
            <Title level={4} style={{ color: '#fa8c16', margin: 0 }}>
              {multifusionInsights.languages_used?.length || 0}
            </Title>
            <Text type="secondary">Ngôn ngữ sử dụng</Text>
            <br />
            <div style={{ marginTop: '8px' }}>
              {multifusionInsights.languages_used?.slice(0, 3).map(lang => (
                <Tag key={lang} size="small">{lang}</Tag>
              ))}
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* Developer Profile Analysis */}
      <Divider />
      <Title level={5}>🎯 Phân tích hồ sơ nhà phát triển</Title>
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <div style={{ padding: '16px', backgroundColor: '#f0f8ff', borderRadius: '8px' }}>
            <Text strong>Phong cách commit: </Text>
            <Tag color="blue">
              {multifusionInsights.productivity_metrics?.avg_changes_per_commit > 100 
                ? '🚀 Commit lớn' 
                : multifusionInsights.productivity_metrics?.avg_changes_per_commit > 50 
                ? '📊 Commit vừa' 
                : '🎯 Commit nhỏ'
              }
            </Tag>
            <br /><br />
            <Text strong>Năng suất: </Text>
            <Tag color="green">
              {multifusionInsights.productivity_metrics?.total_changes > 500 
                ? '💪 Năng suất cao' 
                : multifusionInsights.productivity_metrics?.total_changes > 200 
                ? '📈 Năng suất trung bình' 
                : '🌱 Đang phát triển'
              }
            </Tag>
          </div>
        </Col>
        <Col span={12}>
          <div style={{ padding: '16px', backgroundColor: '#f6fff6', borderRadius: '8px' }}>
            <Text strong>Loại nhà phát triển: </Text>
            <Tag color="purple">
              {(() => {
                const featPercent = (multifusionInsights.commit_type_distribution?.feat || 0) / multifusionInsights.total_commits * 100;
                const fixPercent = (multifusionInsights.commit_type_distribution?.fix || 0) / multifusionInsights.total_commits * 100;
                const refactorPercent = (multifusionInsights.commit_type_distribution?.refactor || 0) / multifusionInsights.total_commits * 100;
                const testPercent = (multifusionInsights.commit_type_distribution?.test || 0) / multifusionInsights.total_commits * 100;
                const docsPercent = (multifusionInsights.commit_type_distribution?.docs || 0) / multifusionInsights.total_commits * 100;
                
                if (featPercent >= 40) return '🚀 Người xây dựng tính năng';
                if (fixPercent >= 30) return '🔧 Người sửa lỗi';
                if (refactorPercent >= 25) return '🛠️ Người tối ưu code';
                if (testPercent >= 20) return '✅ Người đảm bảo chất lượng';
                if (docsPercent >= 15) return '📚 Người viết tài liệu';
                return '🎯 Người đóng góp đa dạng';
              })()}
            </Tag>
            <br /><br />
            <Text strong>Kỹ năng ngôn ngữ: </Text>
            <Tag color="orange">
              {multifusionInsights.languages_used?.length > 3 
                ? '🌐 Đa ngôn ngữ' 
                : multifusionInsights.languages_used?.length === 1 
                ? `🎯 Chuyên ${multifusionInsights.languages_used[0]}` 
                : '⚖️ Tập trung'
              }
            </Tag>
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default MultiFusionInsights;
