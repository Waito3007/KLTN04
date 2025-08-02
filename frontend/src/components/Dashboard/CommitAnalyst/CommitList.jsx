
import React, { useState } from 'react';
import { List, Card, Tag, Input, Select, Button, Typography } from 'antd';

const { Text } = Typography;

const CommitList = ({ commits }) => {
  const [commitSearch, setCommitSearch] = useState('');
  const [commitPage, setCommitPage] = useState(1);
  const [commitPageSize, setCommitPageSize] = useState(10);

  const filterCommits = commits => {
    if (!commitSearch)
      return commits;
    return commits.filter(
      item =>
        (item.message || item.commit?.message || '').toLowerCase().includes(commitSearch.toLowerCase()) ||
        (item.author_name || item.commit?.author?.name || '').toLowerCase().includes(commitSearch.toLowerCase())
    );
  };

  const handleShowDetail = commit => {
    window.alert(`Chi tiết commit: ${commit.message || commit.commit?.message}`);
  };

  const filtered = filterCommits(commits);
  const paged = filtered.slice((commitPage - 1) * commitPageSize, commitPage * commitPageSize);

  return (
    <>
      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <Input
          placeholder="Tìm kiếm commit..."
          value={commitSearch}
          onChange={e => {
            setCommitSearch(e.target.value);
            setCommitPage(1);
          }}
          style={{ width: 220 }}
          allowClear
        />
        <Select
          defaultValue={commitPageSize}
          style={{ width: 120 }}
          onChange={size => {
            setCommitPageSize(size);
            setCommitPage(1);
          }}
          options={[{ value: 5, label: '5 / trang' }, { value: 10, label: '10 / trang' }, { value: 20, label: '20 / trang' }]}
        />
        <Button onClick={() => window.alert('Export danh sách commit!')}>Export</Button>
      </div>
      <List
        itemLayout="vertical"
        dataSource={paged}
        renderItem={item => (
          <List.Item key={item.id || item.sha}>
            <Card size="small" style={{ marginBottom: 8 }}>
              <Text strong>{item.message || item.commit?.message}</Text>
              <div style={{ fontSize: 13, color: '#666', marginTop: 4 }}>
                <Tag color="default">{item.author_name || item.commit?.author?.name}</Tag>
                {item.date && (
                  <span>
                    {' '}
                    | <Tag color="blue">{item.date}</Tag>
                  </span>
                )}
                {item.branch_name && <Tag color="magenta">{item.branch_name}</Tag>}
                {item.analysis && (
                  <>
                    {item.analysis.type && <Tag color="geekblue">{item.analysis.type}</Tag>}
                    {item.analysis.ai_powered && <Tag color="green">AI</Tag>}
                    {typeof item.analysis.confidence === 'number' && (
                      <Tag color="gold">Độ tự tin: {Math.round(item.analysis.confidence * 100)}%</Tag>
                    )}
                    {item.analysis.ai_model && <Tag color="purple">{item.analysis.ai_model}</Tag>}
                  </>
                )}
              </div>
              <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                <Button danger onClick={() => window.alert('Xóa commit này!')}>
                  Xóa
                </Button>
                <Button onClick={() => window.alert('Đánh dấu commit!')}>Đánh dấu</Button>
                <Button type="primary" onClick={() => handleShowDetail(item)}>
                  Chi tiết
                </Button>
              </div>
            </Card>
          </List.Item>
        )}
        pagination={{
          current: commitPage,
          pageSize: commitPageSize,
          total: filtered.length,
          onChange: page => setCommitPage(page),
          showSizeChanger: false,
        }}
      />
    </>
  );
};

export default CommitList;
