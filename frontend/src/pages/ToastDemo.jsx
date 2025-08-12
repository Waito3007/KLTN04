// Demo để test Toast Messages hiển thị đúng overlay
import React from 'react';
import { Button, Space, Divider, Typography } from 'antd';
import Toast from '../components/common/Toast';

const { Title, Text } = Typography;

const ToastDemo = () => {
  // Test basic message toasts
  const testMessageToasts = () => {
    Toast.success('Đây là thông báo thành công!');
    
    setTimeout(() => {
      Toast.error('Đây là thông báo lỗi!');
    }, 500);
    
    setTimeout(() => {
      Toast.warning('Đây là thông báo cảnh báo!');
    }, 1000);
    
    setTimeout(() => {
      Toast.info('Đây là thông báo thông tin!');
    }, 1500);
  };

  // Test notification toasts
  const testNotificationToasts = () => {
    Toast.notify.success({
      message: 'Thành công!',
      description: 'Công việc đã được hoàn thành thành công. Dữ liệu đã được lưu vào hệ thống.',
    });
    
    setTimeout(() => {
      Toast.notify.error({
        message: 'Có lỗi xảy ra!',
        description: 'Không thể kết nối đến server. Vui lòng kiểm tra kết nối mạng và thử lại.',
      });
    }, 500);
    
    setTimeout(() => {
      Toast.notify.warning({
        message: 'Cảnh báo!',
        description: 'Bạn sắp hết dung lượng lưu trữ. Vui lòng dọn dẹp để tránh gián đoạn.',
      });
    }, 1000);
    
    setTimeout(() => {
      Toast.notify.info({
        message: 'Thông tin mới!',
        description: 'Hệ thống sẽ được bảo trì vào lúc 2:00 AM ngày mai. Vui lòng lưu công việc.',
      });
    }, 1500);
  };

  // Test loading và progress
  const testLoadingToasts = () => {
    const loading = Toast.loading('Đang xử lý dữ liệu...');
    
    // Simulate progress
    setTimeout(() => {
      loading.then(() => {
        Toast.success('Xử lý hoàn tất!');
      });
    }, 3000);
    
    // Test progress notification
    const progressNotif = Toast.notify.progress({
      message: 'Đang tải lên tệp...',
      description: 'Vui lòng đợi trong giây lát. Quá trình này có thể mất vài phút.',
    });
    
    setTimeout(() => {
      progressNotif.then(() => {
        Toast.notify.success({
          message: 'Tải lên thành công!',
          description: 'Tệp đã được tải lên và xử lý thành công.',
        });
      });
    }, 4000);
  };

  return (
    <div style={{
      padding: '24px',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <div style={{
        background: 'rgba(255, 255, 255, 0.9)',
        borderRadius: '12px',
        padding: '32px',
        backdropFilter: 'blur(10px)',
        maxWidth: '800px',
        width: '100%',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
      }}>
        <Title level={2} style={{ textAlign: 'center', marginBottom: '32px' }}>
          🎯 Test Toast Messages Overlay
        </Title>
        
        <Text type="secondary" style={{ display: 'block', textAlign: 'center', marginBottom: '32px' }}>
          Kiểm tra xem Toast Messages có hiển thị đúng overlay trên toàn màn hình không
        </Text>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          
          <div>
            <Title level={4}>📝 Message Toasts (Ngắn gọn)</Title>
            <Text type="secondary">Thông báo ngắn, hiển thị ở giữa trên cùng màn hình</Text>
            <br /><br />
            <Button 
              type="primary" 
              size="large"
              onClick={testMessageToasts}
              style={{ width: '100%' }}
            >
              Test Message Toasts (4 loại liên tiếp)
            </Button>
          </div>

          <Divider />

          <div>
            <Title level={4}>🔔 Notification Toasts (Chi tiết)</Title>
            <Text type="secondary">Thông báo chi tiết, hiển thị ở góc phải trên màn hình</Text>
            <br /><br />
            <Button 
              type="primary" 
              size="large"
              onClick={testNotificationToasts}
              style={{ width: '100%' }}
            >
              Test Notification Toasts (4 loại liên tiếp)
            </Button>
          </div>

          <Divider />

          <div>
            <Title level={4}>⏳ Loading & Progress</Title>
            <Text type="secondary">Test loading và progress notifications</Text>
            <br /><br />
            <Button 
              type="primary" 
              size="large"
              onClick={testLoadingToasts}
              style={{ width: '100%' }}
            >
              Test Loading & Progress
            </Button>
          </div>

          <Divider />

          <div>
            <Title level={4}>🧹 Clear All</Title>
            <Text type="secondary">Xóa tất cả toast đang hiển thị</Text>
            <br /><br />
            <Button 
              type="default" 
              size="large"
              onClick={Toast.destroy}
              style={{ width: '100%' }}
            >
              Clear All Toasts
            </Button>
          </div>

        </Space>

        <div style={{ 
          marginTop: '32px', 
          padding: '16px', 
          background: 'rgba(24, 144, 255, 0.1)',
          borderRadius: '8px',
          border: '1px solid rgba(24, 144, 255, 0.2)'
        }}>
          <Title level={5}>✅ Kiểm tra điều kiện:</Title>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            <li>Toast Messages phải hiển thị <strong>TRÊN TOÀN MÀN HÌNH</strong>, không bị giới hạn trong container trắng</li>
            <li>Message toasts xuất hiện ở <strong>giữa trên cùng</strong> màn hình</li>
            <li>Notification toasts xuất hiện ở <strong>góc phải trên</strong> màn hình</li>
            <li>Tất cả toast đều có hiệu ứng <strong>glass morphism</strong> và animation mượt</li>
            <li>Toast có thể <strong>click được</strong> và không bị chặn bởi các element khác</li>
          </ul>
        </div>

      </div>
    </div>
  );
};

export default ToastDemo;
