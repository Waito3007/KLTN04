import React from 'react';
import { Drawer, Space, Tag, Typography, Timeline, Badge } from 'antd';
import { HistoryOutlined } from '@ant-design/icons';

const { Text } = Typography;

const SyncEventsDrawer = ({ eventsDrawerVisible, setEventsDrawerVisible, selectedRepoEvents }) => {
  return (
    <Drawer
      title={
        <Space>
          <HistoryOutlined />
          <span>Sự kiện đồng bộ</span>
          {selectedRepoEvents && (
            <Tag color="blue">{selectedRepoEvents.repo_key}</Tag>
          )}
        </Space>
      }
      placement="right"
      width={500}
      open={eventsDrawerVisible}
      onClose={() => setEventsDrawerVisible(false)}
    >
      {selectedRepoEvents && (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Text strong>Tổng số sự kiện: </Text>
            <Badge count={selectedRepoEvents.total_events} />
          </div>
          
          <Timeline mode="left">
            {selectedRepoEvents?.events?.length > 0 && selectedRepoEvents.events.map((event, index) => {
              const getEventIcon = (eventType) => {
                switch (eventType) {
                  case 'sync_start':
                    return <SyncOutlined style={{ color: '#1890ff' }} />;
                  case 'sync_progress':
                    return <SyncOutlined style={{ color: '#faad14' }} />;
                  case 'sync_complete':
                    return <SyncOutlined style={{ color: '#52c41a' }} />;
                  case 'sync_error':
                    return <SyncOutlined style={{ color: '#ff4d4f' }} />;
                  default:
                    return <SyncOutlined />;
                }
              };

              const getEventColor = (eventType) => {
                switch (eventType) {
                  case 'sync_start':
                    return 'blue';
                  case 'sync_progress':
                    return 'orange';
                  case 'sync_complete':
                    return 'green';
                  case 'sync_error':
                    return 'red';
                  default:
                    return 'gray';
                }
              };

              return (
                <Timeline.Item
                  key={index}
                  dot={getEventIcon(event.event_type)}
                  color={getEventColor(event.event_type)}
                >
                  <div>
                    <div style={{ marginBottom: 4 }}>
                      <Tag color={getEventColor(event.event_type)}>
                        {event.event_type.replace('_', ' ').toUpperCase()}
                      </Tag>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(event.timestamp).toLocaleString()}
                      </Text>
                    </div>
                    
                    {event.data && (
                      <div style={{ marginTop: 8 }}>
                        {event.event_type === 'sync_start' && (
                          <Text>Bắt đầu đồng bộ ({event.data.sync_type})</Text>
                        )}
                        {event.event_type === 'sync_progress' && (
                          <div>
                            <Text style={{ fontSize: '12px' }}>
                              {event.data.current}/{event.data.total} - {event.data.stage}
                            </Text>
                          </div>
                        )}
                        {event.event_type === 'sync_complete' && (
                          <Text type="success">Đồng bộ hoàn thành thành công</Text>
                        )}
                        {event.event_type === 'sync_error' && (
                          <Text type="danger">{event.data.error}</Text>
                        )}
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              );
            })}
          </Timeline>
        </div>
      )}
    </Drawer>
  );
};

export default SyncEventsDrawer;
