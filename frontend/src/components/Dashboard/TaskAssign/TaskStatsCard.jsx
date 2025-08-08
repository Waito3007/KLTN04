/**
 * TaskStatsCard - Card hiển thị thống kê tasks
 * Tuân thủ quy tắc KLTN04: Component đơn giản, reusable
 */

import React, { useMemo } from 'react';
import { Card, Row, Col, Statistic, Progress, Space, Typography } from 'antd';
import { 
  CheckCircleOutlined, 
  ClockCircleOutlined, 
  PlayCircleOutlined,
  StopOutlined 
} from '@ant-design/icons';

const { Text } = Typography;

const TaskStatsCard = ({ stats, loading = false }) => {
  // Defensive programming: Validate stats data
  const validStats = useMemo(() => {
    if (!stats || typeof stats !== 'object') {
      return {
        total: 0,
        todo: 0,
        inProgress: 0,
        done: 0,
        cancelled: 0
      };
    }

    return {
      total: stats.total || 0,
      todo: stats.todo || 0,
      inProgress: stats.inProgress || 0,
      done: stats.done || 0,
      cancelled: stats.cancelled || 0
    };
  }, [stats]);

  // Tính toán progress và tỷ lệ hoàn thành
  const progressData = useMemo(() => {
    const { total, done, inProgress } = validStats;
    
    if (total === 0) {
      return {
        completionRate: 0,
        inProgressRate: 0,
        completionPercent: 0
      };
    }

    const completionRate = (done / total) * 100;
    const inProgressRate = (inProgress / total) * 100;
    
    return {
      completionRate: Math.round(completionRate),
      inProgressRate: Math.round(inProgressRate),
      completionPercent: completionRate
    };
  }, [validStats]);

  // Cấu hình màu sắc cho từng status
  const statusConfigs = {
    todo: {
      icon: ClockCircleOutlined,
      color: '#1890ff',
      title: 'Cần làm'
    },
    inProgress: {
      icon: PlayCircleOutlined,
      color: '#fa8c16',
      title: 'Đang làm'
    },
    done: {
      icon: CheckCircleOutlined,
      color: '#52c41a',
      title: 'Hoàn thành'
    },
    cancelled: {
      icon: StopOutlined,
      color: '#ff4d4f',
      title: 'Đã hủy'
    }
  };

  return (
    <Card 
      title="Thống kê Tasks" 
      size="small" 
      loading={loading}
      className="task-stats-card"
      style={{ marginBottom: 16 }}
    >
      <Row gutter={[16, 16]}>
        {/* Tổng số tasks */}
        <Col xs={12} sm={6}>
          <Statistic
            title="Tổng số Tasks"
            value={validStats.total}
            valueStyle={{ color: '#262626', fontSize: '24px', fontWeight: 'bold' }}
          />
        </Col>

        {/* Tasks theo status */}
        {Object.entries(statusConfigs).map(([key, config]) => {
          const IconComponent = config.icon;
          return (
            <Col xs={12} sm={6} key={key}>
              <Statistic
                title={config.title}
                value={validStats[key]}
                valueStyle={{ color: config.color, fontSize: '20px' }}
                prefix={<IconComponent style={{ color: config.color }} />}
              />
            </Col>
          );
        })}
      </Row>

      {/* Progress Bar */}
      {validStats.total > 0 && (
        <div style={{ marginTop: 16 }}>
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Text strong>Tiến độ hoàn thành</Text>
              <Text>{progressData.completionRate}%</Text>
            </div>
            
            <Progress
              percent={progressData.completionRate}
              status={progressData.completionRate === 100 ? 'success' : 'active'}
              strokeColor={{
                '0%': '#52c41a',
                '100%': '#73d13d',
              }}
              trailColor="#f0f0f0"
            />

            {/* Chi tiết progress */}
            <Row gutter={8} style={{ fontSize: '12px' }}>
              <Col span={8}>
                <Text type="secondary">
                  Hoàn thành: {validStats.done}/{validStats.total}
                </Text>
              </Col>
              <Col span={8}>
                <Text type="secondary">
                  Đang làm: {validStats.inProgress}/{validStats.total}
                </Text>
              </Col>
              <Col span={8}>
                <Text type="secondary">
                  Còn lại: {validStats.todo + validStats.inProgress}/{validStats.total}
                </Text>
              </Col>
            </Row>
          </Space>
        </div>
      )}
    </Card>
  );
};

export default TaskStatsCard;
