import React from 'react';
import { Card } from '@components/common';
import { Space, Progress, Typography } from 'antd';
import { SyncOutlined } from '@ant-design/icons';

const { Text } = Typography;

const GlobalSyncProgress = ({ globalSyncProgress = {
  visible: false,
  totalRepos: 0,
  completedRepos: 0,
  currentRepo: '',
  overallProgress: 0,
} }) => {
  if (!globalSyncProgress.visible) return null;

  return (
    <Card style={{ marginBottom: 24, backgroundColor: '#f6ffed', border: '1px solid #b7eb8f' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text strong style={{ color: '#389e0d' }}>
            <SyncOutlined spin /> Đang đồng bộ tất cả repositories
          </Text>
          <Text style={{ color: '#389e0d' }}>
            {globalSyncProgress.completedRepos}/{globalSyncProgress.totalRepos}
          </Text>
        </div>
        
        <Progress 
          percent={globalSyncProgress.overallProgress} 
          status="active"
          strokeColor={{
            '0%': '#52c41a',
            '100%': '#389e0d',
          }}
          format={(percent) => `${Math.round(percent)}%`}
        />
        
        <Text type="secondary" style={{ fontSize: '12px' }}>
          Hiện tại: {globalSyncProgress.currentRepo}
        </Text>
      </Space>
    </Card>
  );
};

export default GlobalSyncProgress;
