import React from 'react';
import { Card, Space, Select, Switch, Typography, Tag, Button } from 'antd';
import { BranchesOutlined, RobotOutlined } from '@ant-design/icons';

const { Text } = Typography;

/**
 * Component chứa các điều khiển và lựa chọn
 */
const ControlPanel = ({
  branches,
  selectedBranch,
  setSelectedBranch,
  branchesLoading,
  aiModel,
  setAiModel,
  useAI,
  setUseAI,
  aiModelStatus,
  multiFusionV2Status,
  showAIFeatures,
  setShowAIFeatures,
  // NEW PROPS
  fullAnalysisLoading,
  onAnalyzeFullRepo
}) => {
  return (
    <Space wrap>
      {/* Branch Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <BranchesOutlined />
        <Text strong style={{ fontSize: '14px' }}>Nhánh:</Text>
        <Select
          value={selectedBranch}
          onChange={val => setSelectedBranch(val)}
          placeholder="Chọn nhánh"
          style={{ minWidth: 150 }}
          loading={branchesLoading}
          allowClear
        >
          <Select.Option key="all" value={undefined}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Tag color="purple" size="small">Tất cả</Tag>
              Tất cả nhánh
            </span>
          </Select.Option>
          {branches.map(branch => (
            <Select.Option key={branch.name} value={branch.name}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                {branch.is_default && <Tag color="blue" size="small">Mặc định</Tag>}
                {branch.name}
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  ({branch.commits_count} commits)
                </Text>
              </span>
            </Select.Option>
          ))}
        </Select>
      </div>
      
      {/* AI Model Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Text strong>🤖 AI Model:</Text>
        <Select
          value={aiModel}
          onChange={setAiModel}
          style={{ minWidth: 150 }}
          disabled={!useAI}
        >
          <Select.Option value="han" disabled={!aiModelStatus?.model_loaded}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              HAN Model
              {aiModelStatus?.model_loaded ? <Tag color="green" size="small">✅</Tag> : <Tag color="red" size="small">❌</Tag>}
            </span>
          </Select.Option>
          <Select.Option value="multifusion" disabled={!multiFusionV2Status?.model_info?.is_available}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              MultiFusion
              {multiFusionV2Status?.model_info?.is_available ? <Tag color="blue" size="small">✅</Tag> : <Tag color="red" size="small">❌</Tag>}
            </span>
          </Select.Option>
        </Select>
      </div>
      
      {/* AI Toggle Switch */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Text>Phân tích mẫu</Text>
        <Switch 
          checked={useAI}
          onChange={setUseAI}
          checkedChildren="AI"
          unCheckedChildren="Cơ bản"
          style={{
            backgroundColor: useAI ? '#52c41a' : '#d9d9d9'
          }}
        />
        <Text>{aiModel === 'multifusion' ? 'Mô hình MultiFusion' : 'Mô hình HAN AI'}</Text>
      </div>
      
      <Button 
        type="primary" 
        icon={<RobotOutlined />}
        onClick={() => setShowAIFeatures(!showAIFeatures)}
        style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          border: 'none'
        }}
      >
        🤖 AI Features
      </Button>

      {/* NEW: Button for Full Repo Analysis */}
      <Button
        type="default"
        icon={<BranchesOutlined />}
        onClick={onAnalyzeFullRepo}
        loading={fullAnalysisLoading}
        disabled={fullAnalysisLoading}
        style={{
          background: '#f0f2f5',
          borderColor: '#d9d9d9',
          color: 'rgba(0, 0, 0, 0.85)'
        }}
      >
        Phân tích toàn bộ kho lưu trữ
      </Button>
    </Space>
  );
};

export default ControlPanel;
