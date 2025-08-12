// Layout wrapper cho các page để đảm bảo tính nhất quán
import React from 'react';
import { Layout, Breadcrumb } from 'antd';
import { HomeOutlined } from '@ant-design/icons';
import ErrorBoundary from './ErrorBoundary';

const { Content } = Layout;

const PageLayout = ({
  children,
  title,
  breadcrumb = [],
  showBreadcrumb = true,
  background = 'default', // default, gradient, transparent
  padding = 'default', // none, small, default, large
  maxWidth = '1440px',
  centered = false,
  style = {},
  className = '',
  errorBoundary = true
}) => {
  const getBackgroundStyle = () => {
    switch (background) {
      case 'gradient':
        return {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          minHeight: '100vh'
        };
      case 'transparent':
        return {
          background: 'transparent'
        };
      default:
        return {
          background: '#f0f2f5',
          minHeight: '100vh'
        };
    }
  };

  const getPaddingStyle = () => {
    switch (padding) {
      case 'none':
        return { padding: 0 };
      case 'small':
        return { padding: '16px' };
      case 'large':
        return { padding: '48px 24px' };
      default:
        return { padding: '24px' };
    }
  };

  const containerStyle = {
    ...getBackgroundStyle(),
    ...style
  };

  const contentStyle = {
    ...getPaddingStyle(),
    maxWidth: centered ? maxWidth : 'none',
    margin: centered ? '0 auto' : 'initial',
    width: '100%'
  };

  const renderBreadcrumb = () => {
    if (!showBreadcrumb || breadcrumb.length === 0) return null;

    const defaultBreadcrumb = [
      {
        title: <HomeOutlined />,
        href: '/dashboard'
      }
    ];

    const finalBreadcrumb = [...defaultBreadcrumb, ...breadcrumb];

    return (
      <Breadcrumb
        style={{
          marginBottom: '24px',
          padding: '16px 0'
        }}
        items={finalBreadcrumb}
      />
    );
  };

  const renderTitle = () => {
    if (!title) return null;

    return (
      <div style={{
        marginBottom: '24px',
        paddingBottom: '16px',
        borderBottom: '1px solid #f0f0f0'
      }}>
        <h1 style={{
          margin: 0,
          fontSize: '28px',
          fontWeight: '600',
          color: background === 'gradient' ? 'white' : '#262626'
        }}>
          {title}
        </h1>
      </div>
    );
  };

  const content = (
    <Layout style={containerStyle} className={className}>
      <Content style={contentStyle}>
        {renderBreadcrumb()}
        {renderTitle()}
        {children}
      </Content>
    </Layout>
  );

  if (errorBoundary) {
    return (
      <ErrorBoundary>
        {content}
      </ErrorBoundary>
    );
  }

  return content;
};

export default PageLayout;
