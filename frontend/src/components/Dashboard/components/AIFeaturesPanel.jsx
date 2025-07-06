import React from 'react';
import { Card, Row, Col, Typography, Tag, Divider } from 'antd';
import { CodeOutlined, UserOutlined, ToolOutlined, FileTextOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

/**
 * Component hiển thị thông tin về các mô hình AI
 */
const AIFeaturesPanel = ({ aiModelStatus, multiFusionV2Status, useAI, aiModel }) => {
  return (
    <Card 
      style={{ marginBottom: '20px', borderColor: '#1890ff' }}
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>🤖 Trạng thái các mô hình AI</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Tag color={aiModelStatus?.model_loaded ? 'green' : 'red'}>
              🧠 HAN: {aiModelStatus?.model_loaded ? '✅ Đã tải' : '❌ Không khả dụng'}
            </Tag>
            <Tag color={multiFusionV2Status?.model_info?.is_available ? 'blue' : 'red'}>
              🔬 MultiFusion V2: {multiFusionV2Status?.model_info?.is_available ? '✅ Đã tải' : '❌ Không khả dụng'}
            </Tag>
          </div>
        </div>
      }
    >
      <Row gutter={[16, 16]}>
        {/* HAN Model Info */}
        <Col span={12}>
          <Card size="small" title="🧠 Mô hình HAN (Hierarchical Attention Network)">
            <Text strong>Loại mô hình: </Text>
            <Text>{aiModelStatus?.model_info?.type || 'Mạng HAN'}</Text>
            <br />
            <Text strong>Mục đích: </Text>
            <Text>{aiModelStatus?.model_info?.purpose || 'Phân loại commit'}</Text>
            <br />
            <Text strong>Tính năng: </Text>
            <ul style={{ fontSize: '12px', marginTop: '8px' }}>
              {(aiModelStatus?.model_info?.features || []).map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </ul>
          </Card>
        </Col>
        
        {/* MultiFusion V2 Model Info */}
        <Col span={12}>
          <Card size="small" title="🔬 MultiFusion V2 (BERT + MLP Fusion)">
            <Text strong>Kiến trúc: </Text>
            <Text>{multiFusionV2Status?.model_info?.architecture || 'BERT + MLP Fusion'}</Text>
            <br />
            <Text strong>Loại commit hỗ trợ: </Text>
            <Text>{multiFusionV2Status?.model_info?.supported_commit_types?.length || 0} loại</Text>
            <br />
            <Text strong>Ngôn ngữ: </Text>
            <Text>{multiFusionV2Status?.model_info?.supported_languages?.length || 0} ngôn ngữ</Text>
            <ul style={{ fontSize: '12px', marginTop: '8px' }}>
              <li>Phân tích ngữ nghĩa commit (BERT)</li>
              <li>Tích hợp chỉ số code</li>
              <li>Fusion đa tính năng</li>
              <li>Phân loại chính xác cao</li>
              <li>Điểm độ tin cậy</li>
            </ul>
          </Card>
        </Col>
      </Row>
      
      <Divider />
      
      <Text strong>Chế độ hiện tại: </Text>
      <Tag color={useAI ? 'green' : 'orange'}>
        {useAI ? `🤖 Phân tích AI (${aiModel === 'multifusion' ? 'MultiFusion V2' : 'HAN Model'})` : '📝 Phân tích cơ bản'}
      </Tag>
      
      <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
        <Col span={6}>
          <div style={{ textAlign: 'center' }}>
            <CodeOutlined style={{ fontSize: '24px', color: useAI ? '#52c41a' : '#d9d9d9' }} />
            <div>Phân loại Commit</div>
            <Text type="secondary">
              {aiModel === 'multifusion' ? '12 loại (MultiFusion)' : 'feat/fix/chore/docs (HAN)'}
            </Text>
          </div>
        </Col>
        <Col span={6}>
          <div style={{ textAlign: 'center' }}>
            <UserOutlined style={{ fontSize: '24px', color: useAI ? '#1890ff' : '#d9d9d9' }} />
            <div>Phân tích nhà phát triển</div>
            <Text type="secondary">
              {aiModel === 'multifusion' ? 'Hồ sơ nâng cao' : 'Phân tích năng suất'}
            </Text>
          </div>
        </Col>
        <Col span={6}>
          <div style={{ textAlign: 'center' }}>
            <ToolOutlined style={{ fontSize: '24px', color: useAI ? '#fa8c16' : '#d9d9d9' }} />
            <div>Phát hiện tính năng</div>
            <Text type="secondary">
              {aiModel === 'multifusion' ? 'Chỉ số code' : 'Lĩnh vực công nghệ'}
            </Text>
          </div>
        </Col>
        <Col span={6}>
          <div style={{ textAlign: 'center' }}>
            <FileTextOutlined style={{ fontSize: '24px', color: useAI ? '#722ed1' : '#d9d9d9' }} />
            <div>Độ sâu phân tích</div>
            <Text type="secondary">
              {aiModel === 'multifusion' ? 'Fusion đa tính năng' : 'Dựa trên Attention'}
            </Text>
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default AIFeaturesPanel;
