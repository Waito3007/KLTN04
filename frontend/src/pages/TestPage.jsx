import React from 'react';

const TestPage = () => {
  return (
    <div style={{ padding: '20px', margin: '20px' }}>
      <div style={{ padding: '20px', background: 'rgba(24, 144, 255, 0.1)', minHeight: '100vh', borderRadius: '12px' }}>
        <h1>🚀 Test Page - App đang hoạt động!</h1>
        <p>Thời gian: {new Date().toLocaleString()}</p>
        <div>
          <button style={{ 
            padding: '12px 24px', 
            background: 'linear-gradient(45deg, #1890ff, #52c41a)', 
            color: 'white', 
            border: 'none', 
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500'
          }}>
            Click me!
          </button>
        </div>
      </div>
    </div>
  );
};

export default TestPage;
