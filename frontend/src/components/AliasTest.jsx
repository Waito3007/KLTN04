// Test component để kiểm tra alias
import React from 'react';
// Test named imports from index
import { 
  RepoSelector, 
  StatisticsPanel, 
  FiltersPanel 
} from '@taskmanager/index';

const AliasTest = () => {
  console.log('✅ Alias @taskmanager hoạt động!');
  console.log('Components imported:', { RepoSelector, StatisticsPanel, FiltersPanel });
  
  return (
    <div style={{ padding: '20px', border: '2px solid green', borderRadius: '8px' }}>
      <h3>✅ Alias Test thành công!</h3>
      <p>Đã import thành công từ @taskmanager:</p>
      <ul>
        <li>RepoSelector: {RepoSelector ? '✅' : '❌'}</li>
        <li>StatisticsPanel: {StatisticsPanel ? '✅' : '❌'}</li>
        <li>FiltersPanel: {FiltersPanel ? '✅' : '❌'}</li>
      </ul>
    </div>
  );
};

export default AliasTest;
