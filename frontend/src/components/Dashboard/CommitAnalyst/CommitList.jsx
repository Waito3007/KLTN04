
import React, { useState } from 'react';
import { List, Card, Tag, Input, Select, Button, Typography } from 'antd';
import CommitDetailModal from './CommitDetailModal';

const { Text } = Typography;

const COMMIT_TYPES = [
  { value: '', label: 'Tất cả loại' },
  { value: 'feat', label: 'feat' },
  { value: 'fix', label: 'fix' },
  { value: 'docs', label: 'docs' },
  { value: 'chore', label: 'chore' },
  { value: 'style', label: 'style' },
  { value: 'refactor', label: 'refactor' },
  { value: 'test', label: 'test' },
];

const CommitList = ({ commits }) => {
  const [commitSearch, setCommitSearch] = useState('');
  const [commitPage, setCommitPage] = useState(1);
  const [commitPageSize, setCommitPageSize] = useState(10);
  const [commitType, setCommitType] = useState('');
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedCommit, setSelectedCommit] = useState(null);

  const filterCommits = commits => {
    let result = commits;
    if (commitSearch) {
      result = result.filter(
        item =>
          (item.message || item.commit?.message || '').toLowerCase().includes(commitSearch.toLowerCase()) ||
          (item.author_name || item.commit?.author?.name || '').toLowerCase().includes(commitSearch.toLowerCase())
      );
    }
    if (commitType) {
      result = result.filter(
        item => (item.analysis?.type || '').toLowerCase() === commitType
      );
    }
    return result;
  };

  const handleShowDetail = commit => {
    setSelectedCommit(commit);
    setModalVisible(true);
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
          value={commitType}
          style={{ width: 140 }}
          onChange={type => {
            setCommitType(type);
            setCommitPage(1);
          }}
          options={COMMIT_TYPES}
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
      <CommitDetailModal
        visible={modalVisible}
        commit={selectedCommit}
        onClose={() => setModalVisible(false)}
      />
    </>
  );
};

export default CommitList;
