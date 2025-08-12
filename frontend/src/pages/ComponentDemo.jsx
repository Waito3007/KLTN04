// Demo page để showcase các common components
import React, { useState } from 'react';
import { Row, Col, Typography, Space, Button } from 'antd';
import {
  Loading,
  Modal,
  Drawer,
  EmptyState,
  Toast,
  SearchBox
} from '@components/common';

const { Title, Text } = Typography;

const ComponentDemo = () => {
  const [loadingDemo, setLoadingDemo] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [searchValue, setSearchValue] = useState('');
  const [searchFilters, setSearchFilters] = useState({});

  const searchFiltersConfig = [
    {
      key: 'status',
      label: 'Trạng thái',
      options: [
        { value: 'active', label: 'Hoạt động' },
        { value: 'inactive', label: 'Không hoạt động' },
        { value: 'pending', label: 'Chờ xử lý' }
      ]
    },
    {
      key: 'type',
      label: 'Loại',
      options: [
        { value: 'project', label: 'Dự án' },
        { value: 'task', label: 'Nhiệm vụ' },
        { value: 'bug', label: 'Lỗi' }
      ]
    }
  ];

  const handleLoadingDemo = () => {
    setLoadingDemo(true);
    setTimeout(() => {
      setLoadingDemo(false);
      Toast.success('Demo loading hoàn thành!');
    }, 3000);
  };

  const handleToastDemo = (type) => {
    switch (type) {
      case 'success':
        Toast.success('Đây là thông báo thành công!');
        break;
      case 'error':
        Toast.error('Đây là thông báo lỗi!');
        break;
      case 'warning':
        Toast.warning('Đây là thông báo cảnh báo!');
        break;
      case 'info':
        Toast.info('Đây là thông báo thông tin!');
        break;
      case 'loading': {
        const hide = Toast.loading('Đang xử lý...');
        setTimeout(() => {
          hide();
          Toast.success('Xử lý hoàn thành!');
        }, 2000);
        break;
      }
      default:
        break;
    }
  };

  const handleNotificationDemo = (type) => {
    switch (type) {
      case 'success':
        Toast.notify.success({
          message: 'Thành công!',
          description: 'Thao tác đã được thực hiện thành công.',
        });
        break;
      case 'error':
        Toast.notify.error({
          message: 'Lỗi!',
          description: 'Đã xảy ra lỗi trong quá trình xử lý.',
        });
        break;
      case 'warning':
        Toast.notify.warning({
          message: 'Cảnh báo!',
          description: 'Hãy kiểm tra lại thông tin trước khi tiếp tục.',
        });
        break;
      case 'info':
        Toast.notify.info({
          message: 'Thông tin!',
          description: 'Đây là thông tin quan trọng bạn cần biết.',
        });
        break;
      default:
        break;
    }
  };

  return (
    <div style={{
      padding: '24px',
      background: 'rgba(255, 255, 255, 0.9)',
      borderRadius: '16px',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
      margin: '24px'
    }}>
      <Title level={2} style={{
        marginBottom: '8px',
        background: 'linear-gradient(45deg, #1890ff, #52c41a)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        backgroundClip: 'text'
      }}>
        Common Components Demo
      </Title>
      <Text type="secondary" style={{ fontSize: '16px', marginBottom: '32px', display: 'block' }}>
        Showcase của tất cả common components trong ứng dụng
      </Text>

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Loading Components */}
        <div>
          <Title level={3}>Loading Components</Title>
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Button onClick={handleLoadingDemo} loading={loadingDemo} block>
                Demo Loading (3s)
              </Button>
            </Col>
            <Col span={8}>
              <Loading variant="inline" text="Loading inline..." />
            </Col>
            <Col span={8}>
              <Loading size="small" text="Small loading..." />
            </Col>
          </Row>
        </div>

        {/* Modal & Drawer */}
        <div>
          <Title level={3}>Modal & Drawer</Title>
          <Space>
            <Button type="primary" onClick={() => setModalVisible(true)}>
              Mở Modal
            </Button>
            <Button onClick={() => setDrawerVisible(true)}>
              Mở Drawer
            </Button>
          </Space>
        </div>

        {/* Toast Notifications */}
        <div>
          <Title level={3}>Toast Messages</Title>
          <Space wrap>
            <Button onClick={() => handleToastDemo('success')}>Success</Button>
            <Button onClick={() => handleToastDemo('error')}>Error</Button>
            <Button onClick={() => handleToastDemo('warning')}>Warning</Button>
            <Button onClick={() => handleToastDemo('info')}>Info</Button>
            <Button onClick={() => handleToastDemo('loading')}>Loading</Button>
          </Space>
        </div>

        {/* Notifications */}
        <div>
          <Title level={3}>Notification Messages</Title>
          <Space wrap>
            <Button onClick={() => handleNotificationDemo('success')}>Success Notify</Button>
            <Button onClick={() => handleNotificationDemo('error')}>Error Notify</Button>
            <Button onClick={() => handleNotificationDemo('warning')}>Warning Notify</Button>
            <Button onClick={() => handleNotificationDemo('info')}>Info Notify</Button>
          </Space>
        </div>

        {/* Search Box */}
        <div>
          <Title level={3}>Search Box</Title>
          <SearchBox
            placeholder="Tìm kiếm dự án, task..."
            onSearch={setSearchValue}
            onFilter={setSearchFilters}
            searchValue={searchValue}
            filters={searchFiltersConfig}
            activeFilters={searchFilters}
            clearable
          />
          <div style={{ marginTop: '16px' }}>
            <Text>Search value: {searchValue}</Text>
            <br />
            <Text>Active filters: {JSON.stringify(searchFilters)}</Text>
          </div>
        </div>

        {/* Empty States */}
        <div>
          <Title level={3}>Empty States</Title>
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <EmptyState
                type="no-data"
                size="small"
                action="Thêm dữ liệu"
                onAction={() => Toast.info('Thêm dữ liệu clicked!')}
              />
            </Col>
            <Col span={8}>
              <EmptyState
                type="no-search"
                size="small"
              />
            </Col>
            <Col span={8}>
              <EmptyState
                type="error"
                size="small"
                action="Thử lại"
                onAction={() => Toast.info('Thử lại clicked!')}
              />
            </Col>
          </Row>
        </div>
      </Space>

      {/* Modal Demo */}
      <Modal
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
        onConfirm={() => {
          Toast.success('Modal confirmed!');
          setModalVisible(false);
        }}
        title="Demo Modal"
        content="Đây là nội dung của modal demo. Bạn có muốn tiếp tục không?"
        type="info"
      />

      {/* Drawer Demo */}
      <Drawer
        visible={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        title="Demo Drawer"
        showFooter
      >
        <div>
          <Title level={4}>Nội dung Drawer</Title>
          <Text>
            Đây là nội dung của drawer demo. Bạn có thể đặt bất kỳ component nào ở đây.
          </Text>
          <div style={{ marginTop: '16px' }}>
            <EmptyState
              type="folder"
              size="small"
              title="Thư mục trống"
              description="Chưa có file nào trong thư mục này."
            />
          </div>
        </div>
      </Drawer>

      {/* Loading overlay when demo is running */}
      {loadingDemo && (
        <Loading
          variant="overlay"
          text="Đang chạy demo..."
          size="large"
        />
      )}
    </div>
  );
};

export default ComponentDemo;