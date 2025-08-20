import React from 'react';
import { Tabs, Table, Badge } from 'antd';

const SyncTabs = ({ activeTab, setActiveTab, syncStatus, getColumns }) => {
  const getTabCount = (type) => {
    if (!syncStatus) return 0;
    return syncStatus.repositories[type]?.length || 0;
  };

  const getTabItems = () => [
    {
      key: 'unsynced',
      label: (
        <Badge count={getTabCount('unsynced')} offset={[10, 0]}>
          <span>Chưa đồng bộ</span>
        </Badge>
      ),
      children: (
        <Table
          columns={safeGetColumns()}
          dataSource={syncStatus?.repositories?.unsynced || []}
          rowKey={(record) => `${record.owner}-${record.name}`}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      ),
    },
    {
      key: 'outdated',
      label: (
        <Badge count={getTabCount('outdated')} offset={[10, 0]}>
          <span>Cần cập nhật</span>
        </Badge>
      ),
      children: (
        <Table
          columns={safeGetColumns()}
          dataSource={syncStatus?.repositories?.outdated || []}
          rowKey={(record) => `${record.owner}-${record.name}`}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      ),
    },
    {
      key: 'synced',
      label: (
        <Badge count={getTabCount('synced')} offset={[10, 0]}>
          <span>Đã đồng bộ</span>
        </Badge>
      ),
      children: (
        <Table
          columns={safeGetColumns()}
          dataSource={syncStatus?.repositories?.synced || []}
          rowKey={(record) => `${record.owner}-${record.name}`}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      ),
    },
  ];

  const safeGetColumns = () => {
    const columns = getColumns();
    if (!Array.isArray(columns)) {
      console.error('getColumns did not return an array:', columns);
      return [];
    }
    return columns;
  };

  return (
    <Tabs 
      activeKey={activeTab} 
      onChange={setActiveTab}
      items={getTabItems()}
    />
  );
};

export default SyncTabs;
