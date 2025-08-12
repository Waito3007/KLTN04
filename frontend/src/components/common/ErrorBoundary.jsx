// Error Boundary component để bắt lỗi React
import React from 'react';
import { Result, Button, Typography, Card, Space } from 'antd';
import { BugOutlined, ReloadOutlined, HomeOutlined } from '@ant-design/icons';
import theme from './theme';

const { Text, Paragraph } = Typography;

const ErrorFallback = ({ 
  error, 
  resetError,
  showDetails = false,
  title = 'Oops! Có lỗi xảy ra',
  subTitle = 'Đã xảy ra lỗi không mong muốn. Vui lòng thử lại hoặc liên hệ hỗ trợ.',
}) => {
  const handleReload = () => {
    window.location.reload();
  };

  const handleGoHome = () => {
    window.location.href = '/dashboard';
  };

  return (
    <div style={{
      minHeight: '70vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: theme.colors.gradient.subtle,
      padding: theme.spacing.xl,
    }}>
      <Card
        style={{
          maxWidth: 600,
          width: '100%',
          borderRadius: theme.borderRadius.xl,
          boxShadow: theme.shadows.xl,
          border: 'none',
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(10px)',
        }}
        styles={{
          body: { 
            padding: theme.spacing.xxl,
            textAlign: 'center'
          }
        }}
      >
        <Result
          icon={
            <div style={{
              background: theme.colors.gradient.danger,
              borderRadius: theme.borderRadius.round,
              width: 80,
              height: 80,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto',
              marginBottom: theme.spacing.lg,
              boxShadow: theme.shadows.glow,
              animation: 'pulse 2s infinite',
            }}>
              <BugOutlined style={{ 
                fontSize: 36, 
                color: theme.colors.white 
              }} />
            </div>
          }
          title={
            <Text style={{ 
              fontSize: theme.fontSizes.xxl,
              fontWeight: theme.fontWeights.bold,
              background: theme.colors.gradient.primary,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              display: 'block',
              marginBottom: theme.spacing.md,
            }}>
              {title}
            </Text>
          }
          subTitle={
            <div style={{ marginTop: theme.spacing.md }}>
              <Paragraph style={{ 
                color: theme.colors.text.secondary,
                fontSize: theme.fontSizes.md,
                lineHeight: 1.6,
                marginBottom: theme.spacing.lg,
              }}>
                {subTitle}
              </Paragraph>
              
              {showDetails && error && (
                <Card 
                  size="small"
                  style={{
                    background: '#fff2f0',
                    border: '1px solid #ffccc7',
                    borderRadius: theme.borderRadius.md,
                    marginBottom: theme.spacing.lg,
                    textAlign: 'left',
                  }}
                >
                  <Text strong style={{ 
                    color: '#cf1322',
                    fontSize: theme.fontSizes.sm 
                  }}>
                    Chi tiết lỗi (Development):
                  </Text>
                  <pre style={{
                    marginTop: theme.spacing.sm,
                    fontSize: theme.fontSizes.xs,
                    color: theme.colors.text.tertiary,
                    overflow: 'auto',
                    maxHeight: 200,
                    background: '#fafafa',
                    padding: theme.spacing.sm,
                    borderRadius: theme.borderRadius.sm,
                    border: '1px solid #f0f0f0',
                  }}>
                    {error.stack}
                  </pre>
                </Card>
              )}
            </div>
          }
          extra={
            <Space size="large" direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Button 
                  type="primary" 
                  icon={<ReloadOutlined />}
                  onClick={resetError || handleReload}
                  size="large"
                  style={{
                    background: theme.colors.gradient.primary,
                    border: 'none',
                    borderRadius: theme.borderRadius.lg,
                    fontWeight: theme.fontWeights.medium,
                    height: 48,
                    paddingLeft: theme.spacing.xl,
                    paddingRight: theme.spacing.xl,
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
                  Thử lại
                </Button>
                
                <Button 
                  icon={<HomeOutlined />}
                  onClick={handleGoHome}
                  size="large"
                  style={{
                    borderRadius: theme.borderRadius.lg,
                    fontWeight: theme.fontWeights.medium,
                    height: 48,
                    paddingLeft: theme.spacing.xl,
                    paddingRight: theme.spacing.xl,
                    borderColor: theme.colors.border.default,
                    transition: theme.transitions.normal,
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = 'translateY(-2px)';
                    e.target.style.borderColor = theme.colors.primary;
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.borderColor = theme.colors.border.default;
                  }}
                >
                  Về trang chủ
                </Button>
              </Space>
              
              <Text style={{ 
                color: theme.colors.text.tertiary,
                fontSize: theme.fontSizes.sm 
              }}>
                Nếu vấn đề vẫn tiếp tục, vui lòng liên hệ hỗ trợ kỹ thuật
              </Text>
            </Space>
          }
        />
      </Card>

      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes pulse {
            0% {
              transform: scale(1);
            }
            50% {
              transform: scale(1.05);
            }
            100% {
              transform: scale(1);
            }
          }
        `
      }} />
    </div>
  );
};

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    
    // Có thể gửi lỗi lên logging service
    if (import.meta.env.PROD) {
      // Gửi lỗi lên service tracking như Sentry
      console.log('Would send error to tracking service');
    }
  }

  resetError = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <ErrorFallback 
          error={this.state.error}
          resetError={this.resetError}
          showDetails={import.meta.env.DEV}
          {...this.props}
        />
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
export { ErrorFallback };
