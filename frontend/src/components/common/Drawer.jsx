// Drawer component với thiết kế hiện đại
import React from 'react';
import { Drawer, Typography, Button, Space } from 'antd';
import { CloseOutlined } from '@ant-design/icons';

const { Title } = Typography;

const CustomDrawer = ({
  visible = false,
  onClose,
  title = 'Drawer',
  placement = 'right', // left, right, top, bottom
  size = 'default', // small, default, large
  showFooter = false,
  footerActions = null,
  headerStyle = {},
  bodyStyle = {},
  children,
  ...props
}) => {
  const getWidth = () => {
    if (placement === 'top' || placement === 'bottom') {
      return undefined; // Sử dụng height cho top/bottom
    }
    
    switch (size) {
      case 'small':
        return 400;
      case 'large':
        return 736;
      default:
        return 520;
    }
  };

  const getHeight = () => {
    if (placement === 'left' || placement === 'right') {
      return undefined; // Sử dụng width cho left/right
    }
    
    switch (size) {
      case 'small':
        return 300;
      case 'large':
        return 600;
      default:
        return 400;
    }
  };

  const getHeaderStyle = () => {
    return {
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderBottom: 'none',
      padding: '16px 24px',
      ...headerStyle
    };
  };

  const getBodyStyle = () => {
    return {
      background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      padding: '24px',
      ...bodyStyle
    };
  };

  const renderFooter = () => {
    if (!showFooter) return null;

    const defaultActions = (
      <Space>
        <Button onClick={onClose}>
          Đóng
        </Button>
        <Button 
          type="primary"
          style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            border: 'none',
            borderRadius: '8px',
          }}
        >
          Xác nhận
        </Button>
      </Space>
    );

    return (
      <div style={{
        padding: '16px 24px',
        background: '#fafafa',
        borderTop: '1px solid #f0f0f0',
        display: 'flex',
        justifyContent: 'flex-end'
      }}>
        {footerActions || defaultActions}
      </div>
    );
  };

  return (
    <Drawer
      title={
        <Title 
          level={4} 
          style={{ 
            color: 'white', 
            margin: 0,
            fontSize: '18px',
            fontWeight: '600'
          }}
        >
          {title}
        </Title>
      }
      placement={placement}
      width={getWidth()}
      height={getHeight()}
      onClose={onClose}
      open={visible}
      styles={{
        header: getHeaderStyle(),
        body: getBodyStyle(),
      }}
      closeIcon={
        <CloseOutlined 
          style={{ 
            color: 'white', 
            fontSize: '16px',
            padding: '4px'
          }} 
        />
      }
      footer={renderFooter()}
      {...props}
    >
      {children}
    </Drawer>
  );
};

export default CustomDrawer;
