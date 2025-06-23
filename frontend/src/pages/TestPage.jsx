import React from 'react';

const TestPage = () => {
  return (
    <div style={{ padding: '20px', background: 'lightblue', minHeight: '100vh' }}>
      <h1>🚀 Test Page - App đang hoạt động!</h1>
      <p>Thời gian: {new Date().toLocaleString()}</p>
      <div>
        <button style={{ padding: '10px', background: 'green', color: 'white', border: 'none', borderRadius: '5px' }}>
          Click me!
        </button>
      </div>
    </div>
  );
};

export default TestPage;
