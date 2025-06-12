import React, { useState, useEffect, useLayoutEffect } from 'react';
import { Progress, Card, Typography, Button, Space } from 'antd';
import { CloseOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Text } = Typography;

const ProgressContainer = styled(Card)`
  position: fixed;
  top: 80px;
  right: 20px;
  width: 320px;
  z-index: 1000;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border-radius: 8px;
  opacity: ${props => props.visible ? 1 : 0};
  transform: ${props => props.visible ? 'translateX(0)' : 'translateX(100%)'};
  transition: opacity 0.1s ease-out, transform 0.1s ease-out;
  pointer-events: ${props => props.visible ? 'auto' : 'none'};

  .ant-card-body {
    padding: 16px;
  }

  /* Force immediate display */
  &.instant-show {
    opacity: 1 !important;
    transform: translateX(0) !important;
    transition: none !important;
  }
`;

const ProgressHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
`;

const ProgressTitle = styled(Text)`
  font-weight: 600;
  color: #1e293b;
`;

const RepoProgress = styled.div`
  margin-bottom: 8px;
  padding: 8px;
  background: #f8fafc;
  border-radius: 6px;
  border-left: 3px solid ${props => 
    props.status === 'completed' ? '#10b981' : 
    props.status === 'error' ? '#ef4444' : '#3b82f6'
  };
`;

const SyncProgressNotification = ({ 
  visible, 
  onClose, 
  totalRepos = 0, 
  completedRepos = 0, 
  currentRepo = '', 
  repoProgresses = [], 
  overallProgress = 0 
}) => {
  const [autoClose, setAutoClose] = useState(false);
  const [showInstantly, setShowInstantly] = useState(false);
  const [forceInstantShow, setForceInstantShow] = useState(false);
  // Show immediately when visible becomes true - using useLayoutEffect for immediate DOM update
  useLayoutEffect(() => {
    if (visible) {
      setShowInstantly(true);
      setForceInstantShow(true);
    }
  }, [visible]);

  useEffect(() => {
    if (visible) {
      // Reset force instant after a tiny delay to allow normal transitions
      const timer = setTimeout(() => setForceInstantShow(false), 50);
      return () => clearTimeout(timer);
    } else {
      setForceInstantShow(false);
      // Delay hiding for animation
      const timer = setTimeout(() => setShowInstantly(false), 200);
      return () => clearTimeout(timer);
    }
  }, [visible]);

  useEffect(() => {
    if (completedRepos === totalRepos && totalRepos > 0) {
      setAutoClose(true);
      const timer = setTimeout(() => {
        onClose();
      }, 3000); // Tự động đóng sau 3 giây
      return () => clearTimeout(timer);
    }
  }, [completedRepos, totalRepos, onClose]);
  
  // Render even if not visible for smooth transitions
  if (!showInstantly && !visible) return null;

  const isCompleted = completedRepos === totalRepos && totalRepos > 0;
  const hasErrors = repoProgresses.some(repo => repo.status === 'error');
  return (
    <ProgressContainer 
      visible={visible} 
      className={forceInstantShow ? 'instant-show' : (visible ? 'show' : '')}
    >
      <ProgressHeader>
        <ProgressTitle>
          {isCompleted ? '✅ Đồng bộ hoàn thành' : '🔄 Đang đồng bộ repository'}
        </ProgressTitle>
        <Button 
          type="text" 
          size="small" 
          icon={<CloseOutlined />} 
          onClick={onClose}
        />
      </ProgressHeader>

      {/* Overall Progress */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <Text type="secondary">Tổng tiến trình</Text>
          <Text strong>{Math.round(overallProgress)}%</Text>
        </div>
        <Progress 
          percent={overallProgress} 
          strokeColor={isCompleted ? '#10b981' : '#3b82f6'}
          showInfo={false}
          size="small"
        />
        <Text type="secondary" style={{ fontSize: '12px' }}>
          {completedRepos}/{totalRepos} repository
        </Text>
      </div>

      {/* Current Repository */}
      {currentRepo && !isCompleted && (
        <div style={{ marginBottom: 12 }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>Đang xử lý:</Text>
          <div style={{ 
            background: '#e0f2fe', 
            padding: '4px 8px', 
            borderRadius: '4px',
            marginTop: '4px'
          }}>
            <Text style={{ fontSize: '12px', color: '#0369a1' }}>{currentRepo}</Text>
          </div>
        </div>
      )}

      {/* Repository List (hiển thị khi có nhiều repo) */}
      {repoProgresses.length > 0 && (
        <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
          {repoProgresses.slice(-5).map((repo, index) => (
            <RepoProgress key={index} status={repo.status}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text style={{ fontSize: '11px', fontWeight: 500 }}>
                  {repo.name}
                </Text>
                <Space size={4}>
                  {repo.status === 'completed' && <CheckCircleOutlined style={{ color: '#10b981' }} />}
                  {repo.status === 'error' && <ExclamationCircleOutlined style={{ color: '#ef4444' }} />}
                  <Text style={{ fontSize: '10px' }}>
                    {repo.status === 'completed' ? '✓' : 
                     repo.status === 'error' ? '✗' : '...'}
                  </Text>
                </Space>
              </div>
              {repo.progress !== undefined && repo.status === 'syncing' && (
                <Progress 
                  percent={repo.progress} 
                  size="small" 
                  showInfo={false}
                  strokeColor="#3b82f6"
                  style={{ marginTop: 4 }}
                />
              )}
            </RepoProgress>
          ))}
        </div>
      )}

      {/* Summary */}
      {isCompleted && (
        <div style={{ 
          background: hasErrors ? '#fef3c7' : '#d1fae5', 
          padding: '8px', 
          borderRadius: '6px',
          marginTop: '12px'
        }}>
          <Text style={{ 
            fontSize: '12px', 
            color: hasErrors ? '#92400e' : '#047857'
          }}>
            {hasErrors 
              ? `Hoàn thành với ${repoProgresses.filter(r => r.status === 'error').length} lỗi`
              : 'Tất cả repository đã được đồng bộ thành công!'
            }
          </Text>
          {autoClose && (
            <div style={{ marginTop: 4 }}>
              <Text style={{ fontSize: '10px', color: '#6b7280' }}>
                Tự động đóng sau 3 giây...
              </Text>
            </div>
          )}
        </div>
      )}
    </ProgressContainer>
  );
};

export default SyncProgressNotification;
