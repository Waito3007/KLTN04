// Simple test for alias
import React from 'react';

// Test individual imports
import RepoSelector from '@taskmanager/RepoSelector';
import StatisticsPanel from '@taskmanager/StatisticsPanel';

const SimpleAliasTest = () => {
  console.log('Testing individual imports:', { RepoSelector, StatisticsPanel });
  
  return (
    <div style={{ padding: '10px', background: '#f0f0f0', margin: '10px' }}>
      <h4>Simple Alias Test</h4>
      <p>RepoSelector: {RepoSelector ? '✅ Loaded' : '❌ Failed'}</p>
      <p>StatisticsPanel: {StatisticsPanel ? '✅ Loaded' : '❌ Failed'}</p>
    </div>
  );
};

export default SimpleAliasTest;
