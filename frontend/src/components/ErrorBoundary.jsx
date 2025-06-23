// frontend/src/components/ErrorBoundary.jsx
import React from 'react';
import { Alert, Button } from 'antd';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '40px', 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          minHeight: '100vh',
          backgroundColor: '#f5f5f5'
        }}>
          <div style={{ maxWidth: '600px', width: '100%' }}>
            <Alert
              message="Đã xảy ra lỗi!"
              description={
                <div>
                  <p>Ứng dụng đã gặp lỗi không mong muốn. Vui lòng thử lại hoặc liên hệ hỗ trợ.</p>
                  <details style={{ marginTop: '16px' }}>
                    <summary>Chi tiết lỗi (cho developer)</summary>
                    <pre style={{ 
                      marginTop: '8px', 
                      padding: '8px', 
                      backgroundColor: '#f8f8f8',
                      borderRadius: '4px',
                      fontSize: '12px',
                      overflow: 'auto'
                    }}>
                      {this.state.error?.toString()}
                    </pre>
                  </details>
                </div>
              }
              type="error"
              showIcon
              action={
                <Button 
                  size="small" 
                  danger 
                  onClick={() => window.location.reload()}
                >
                  Tải lại trang
                </Button>
              }
            />
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
