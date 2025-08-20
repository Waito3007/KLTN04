import React from 'react';
import { Modal, Space, Typography, Select } from 'antd';
import { SyncOutlined } from '@ant-design/icons';

const { Text } = Typography;
const { Option } = Select;

const SyncModal = ({ syncModalVisible, setSyncModalVisible, selectedRepo, syncType, setSyncType, handleModalSync }) => {
  return (
    <Modal
      title="Tùy chọn đồng bộ"
      open={syncModalVisible}
      onOk={handleModalSync}
      onCancel={() => setSyncModalVisible(false)}
      okText="Đồng bộ"
      cancelText="Hủy"
    >
      {selectedRepo && (
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Repository: </Text>
            <Text>{selectedRepo.full_name || selectedRepo.name}</Text>
          </div>
          
          <div>
            <Text strong>Loại đồng bộ: </Text>
            <Select 
              value={syncType} 
              onChange={setSyncType}
              style={{ width: '100%', marginTop: 8 }}
            >
              <Option value="basic">
                <Space>
                  <SyncOutlined />
                  <span>Cơ bản (Repository + Branches)</span>
                </Space>
              </Option>
              <Option value="enhanced">
                <Space>
                  <SyncOutlined />
                  <span>Nâng cao (+ Commits + Issues + PRs)</span>
                </Space>
              </Option>
              <Option value="optimized">
                <Space>
                  <SyncOutlined />
                  <span>Tối ưu (Background + Concurrent + Diff)</span>
                </Space>
              </Option>
            </Select>
          </div>

          <div style={{ marginTop: 16 }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {syncType === 'basic' && 'Đồng bộ thông tin repository và branches cơ bản'}
              {syncType === 'enhanced' && 'Đồng bộ đầy đủ commits, issues và pull requests'}
              {syncType === 'optimized' && 'Đồng bộ background với tốc độ cao nhất, bao gồm code diff'}
            </Text>
          </div>
        </Space>
      )}
    </Modal>
  );
};

export default SyncModal;
