import React from 'react';
import { Table } from 'antd';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#3b82f6', '#6366f1', '#10b981', '#f59e42', '#ef4444', '#a855f7', '#f43f5e'];

const CommitTypeChart = ({ commitTypes = {}, totalCommits = 0 }) => {
  const data = Object.entries(commitTypes).map(([type, count], idx) => ({
    type,
    count,
    color: COLORS[idx % COLORS.length]
  }));

  const columns = [
    {
      title: 'Loại commit',
      dataIndex: 'type',
      key: 'type',
      render: (text, record) => <span style={{ color: record.color, fontWeight: 600 }}>{text}</span>
    },
    {
      title: 'Số lượng',
      dataIndex: 'count',
      key: 'count',
      align: 'center',
      render: (count) => <span style={{ fontWeight: 500 }}>{count}</span>
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <Table
        columns={columns}
        dataSource={data}
        pagination={false}
        size="small"
        rowKey="type"
        style={{ marginBottom: 16, borderRadius: 12, boxShadow: '0 2px 8px rgba(59,130,246,0.08)' }}
      />
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              dataKey="count"
              nameKey="type"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={({ type, percent }) => `${type}: ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry, idx) => (
                <Cell key={`cell-${idx}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div style={{ textAlign: 'center', marginTop: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 16 }}>Tổng số commit: {totalCommits}</span>
      </div>
    </div>
  );
};

export default CommitTypeChart;
