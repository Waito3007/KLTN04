// Empty State component cho các trạng thái trống
import React from 'react';
import { Empty, Button, Typography, Card } from 'antd';
import { 
  FolderOpenOutlined, 
  FileTextOutlined, 
  DatabaseOutlined,
  SearchOutlined,
  WifiOutlined,
  ExclamationCircleOutlined 
} from '@ant-design/icons';
import theme from './theme';

const { Text, Title } = Typography;

const EmptyState = ({
  type = 'default', // default, no-data, no-search, no-connection, error, folder
  title,
  description,
  action,
  actionText = 'Thực hiện',
  onAction,
  style = {},
  size = 'default' // small, default, large
}) => {
  const getIcon = () => {
    const iconStyle = {
      fontSize: size === 'large' ? '80px' : size === 'small' ? '40px' : '60px',
      color: theme.colors.text.tertiary,
      opacity: 0.6,
    };

    switch (type) {
      case 'no-data':
        return <DatabaseOutlined style={{ ...iconStyle, color: theme.colors.techBlue }} />;
      case 'no-search':
        return <SearchOutlined style={{ ...iconStyle, color: theme.colors.techPurple }} />;
      case 'no-connection':
        return <WifiOutlined style={{ ...iconStyle, color: theme.colors.danger }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ ...iconStyle, color: theme.colors.warning }} />;
      case 'folder':
        return <FolderOpenOutlined style={{ ...iconStyle, color: theme.colors.techCyan }} />;
      default:
        return <FileTextOutlined style={{ ...iconStyle, color: theme.colors.techGreen }} />;
    }
  };

  const getDefaultContent = () => {
    switch (type) {
      case 'no-data':
        return {
          title: 'Không có dữ liệu',
          description: 'Hiện tại chưa có dữ liệu nào được tìm thấy.'
        };
      case 'no-search':
        return {
          title: 'Không tìm thấy kết quả',
          description: 'Thử tìm kiếm với từ khóa khác hoặc điều chỉnh bộ lọc.'
        };
      case 'no-connection':
        return {
          title: 'Mất kết nối',
          description: 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng.'
        };
      case 'error':
        return {
          title: 'Có lỗi xảy ra',
          description: 'Đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại.'
        };
      case 'folder':
        return {
          title: 'Thư mục trống',
          description: 'Thư mục này chưa có tệp nào.'
        };
      default:
        return {
          title: 'Trống',
          description: 'Chưa có nội dung nào.'
        };
    }
  };

  const defaultContent = getDefaultContent();
  const finalTitle = title || defaultContent.title;
  const finalDescription = description || defaultContent.description;

  const containerStyle = {
    padding: size === 'large' ? theme.spacing.xxxl : size === 'small' ? theme.spacing.xl : theme.spacing.xxl,
    textAlign: 'center',
    background: theme.colors.gradient.subtle,
    borderRadius: theme.borderRadius.xl,
    border: `1px solid ${theme.colors.border.light}`,
    minHeight: '200px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ...style
  };

  return (
    <div style={containerStyle}>
      <Empty
        image={
          <div style={{
            background: theme.colors.bg.glass,
            borderRadius: theme.borderRadius.round,
            width: size === 'large' ? 120 : size === 'small' ? 80 : 100,
            height: size === 'large' ? 120 : size === 'small' ? 80 : 100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto',
            marginBottom: theme.spacing.lg,
            boxShadow: theme.shadows.md,
            backdropFilter: 'blur(10px)',
          }}>
            {getIcon()}
          </div>
        }
        description={
          <div>
            <Title 
              level={size === 'large' ? 3 : size === 'small' ? 5 : 4}
              style={{ 
                color: theme.colors.text.primary,
                marginBottom: theme.spacing.sm,
                fontWeight: theme.fontWeights.semibold,
              }}
            >
              {finalTitle}
            </Title>
            <Text style={{ 
              fontSize: size === 'large' ? theme.fontSizes.md : size === 'small' ? theme.fontSizes.xs : theme.fontSizes.sm,
              color: theme.colors.text.secondary,
              lineHeight: 1.6,
            }}>
              {finalDescription}
            </Text>
          </div>
        }
      >
        {(action || onAction) && (
          <Button 
            type="primary"
            onClick={onAction}
            size="large"
            style={{
              marginTop: theme.spacing.lg,
              background: theme.colors.gradient.primary,
              border: 'none',
              borderRadius: theme.borderRadius.lg,
              height: 48,
              paddingLeft: theme.spacing.xl,
              paddingRight: theme.spacing.xl,
              fontWeight: theme.fontWeights.medium,
              boxShadow: theme.shadows.md,
              transition: theme.transitions.normal,
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = theme.shadows.glowHover;
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = theme.shadows.md;
            }}
          >
            {action || actionText}
          </Button>
        )}
      </Empty>
    </div>
  );
};

export default EmptyState;
