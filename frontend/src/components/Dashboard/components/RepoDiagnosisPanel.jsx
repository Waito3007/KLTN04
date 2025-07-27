import React, { useEffect, useState } from 'react';
import { Card, Typography, Spin, Empty, Tag, Divider, List, Select, Input, Avatar } from 'antd';
import FadeInWrapper from './FadeInWrapper';
import CommitTypeChart from './CommitTypeChart';
import CommitDetailModal from './CommitDetailModal';

const { Title, Text } = Typography;
const { Option } = Select;

// Modern, unified diagnosis panel for all repo analyses
const RepoDiagnosisPanel = ({ repositories = [], onRepoChange }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [commitAnalysis, setCommitAnalysis] = useState(null);
  const [areaAnalysis, setAreaAnalysis] = useState(null);
  const [riskAnalysis, setRiskAnalysis] = useState(null);
  const [allCommits, setAllCommits] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [repoId, setRepoId] = useState(null);

  // Filtered repo list for search
  const filteredRepos = repositories.filter(repo =>
    repo.name.toLowerCase().includes(searchText.toLowerCase()) ||
    repo.owner?.login?.toLowerCase().includes(searchText.toLowerCase())
  );

  useEffect(() => {
    // Auto-select first repo if available
    if (repositories.length > 0 && !repoId) {
      setRepoId(repositories[0].id);
      if (onRepoChange) onRepoChange(repositories[0]);
    }
  }, [repositories]);

  // Fetch analysis data when repoId changes
  useEffect(() => {
    if (!repoId) {
      setCommitAnalysis(null);
      setAreaAnalysis(null);
      setRiskAnalysis(null);
      setAllCommits(null);
      return;
    }
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all/analysis`).then(r => r.ok ? r.json() : null),
      fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all`).then(r => r.ok ? r.json() : null),
      fetch(`http://localhost:8000/api/area-analysis/repositories/${repoId}/full-area-analysis`).then(r => r.ok ? r.json() : null),
      fetch(`http://localhost:8000/api/risk-analysis/repositories/${repoId}/full-risk-analysis`).then(r => r.ok ? r.json() : null)
    ]).then(([analystData, allData, areaData, riskData]) => {
      setCommitAnalysis(analystData);
      setAreaAnalysis(areaData);
      setRiskAnalysis(riskData);
      setAllCommits(allData);
    }).catch(() => {
      setError('Lỗi khi tải dữ liệu phân tích kho lưu trữ.');
    }).finally(() => {
      setLoading(false);
    });
  }, [repoId]);

  // Handle repo selection
  const handleRepoSelect = (id) => {
    setRepoId(id);
    if (onRepoChange) {
      const repo = repositories.find(r => r.id === id);
      onRepoChange(repo);
    }
  };

  // State for commit detail modal
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [detailCommit, setDetailCommit] = useState(null);
  const [detailFiles, setDetailFiles] = useState([]);
  const [detailDiffs, setDetailDiffs] = useState({});

  // Handler for showing commit detail
  const handleShowDetail = async (commit) => {
    setDetailCommit(commit);
    setDetailModalOpen(true);
    // TODO: Fetch files and diffs for commit
    // Demo: fake data
    setDetailFiles(commit.files || [
      { filename: 'src/App.js', status: 'modified', language: 'javascript' },
      { filename: 'README.md', status: 'added', language: 'markdown' }
    ]);
    setDetailDiffs({
      'src/App.js': `-console.log('Hello')\n+console.log('Hello World!')`,
      'README.md': `+# Project\n+This is a new README file.`
    });
  };

  // Helper for commit list rendering with pagination and search/filter
  const [commitSearch, setCommitSearch] = useState('');
  const [commitPage, setCommitPage] = useState(1);
  const [commitPageSize, setCommitPageSize] = useState(10);

  const filterCommits = (commits) => {
    if (!commitSearch) return commits;
    return commits.filter(item =>
      (item.message || item.commit?.message || '').toLowerCase().includes(commitSearch.toLowerCase()) ||
      (item.author_name || item.commit?.author?.name || '').toLowerCase().includes(commitSearch.toLowerCase())
    );
  };

  const renderCommitList = (commits) => {
    const filtered = filterCommits(commits);
    const paged = filtered.slice((commitPage - 1) * commitPageSize, commitPage * commitPageSize);
    return (
      <>
        <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
          <Input
            placeholder="Tìm kiếm commit..."
            value={commitSearch}
            onChange={e => { setCommitSearch(e.target.value); setCommitPage(1); }}
            style={{ width: 220 }}
            allowClear
          />
          {/* Quản lý commit: ví dụ export, xóa, đánh dấu... */}
          <Select
            defaultValue={commitPageSize}
            style={{ width: 120 }}
            onChange={size => { setCommitPageSize(size); setCommitPage(1); }}
            options={[{ value: 5, label: '5 / trang' }, { value: 10, label: '10 / trang' }, { value: 20, label: '20 / trang' }]}
          />
          <button style={{ border: 'none', background: '#3b82f6', color: 'white', borderRadius: 6, padding: '0 12px', cursor: 'pointer' }} onClick={() => window.alert('Export danh sách commit!')}>Export</button>
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
                  {item.date && <span> | <Tag color="blue">{item.date}</Tag></span>}
                  {item.analysis && item.analysis.type && <Tag color="geekblue">{item.analysis.type}</Tag>}
                  {item.analysis && item.analysis.ai_powered && <Tag color="green">AI</Tag>}
                </div>
                {/* Quản lý commit: ví dụ xóa, đánh dấu, xem chi tiết... */}
                <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                  <button style={{ border: 'none', background: '#ef4444', color: 'white', borderRadius: 6, padding: '0 10px', cursor: 'pointer' }} onClick={() => window.alert('Xóa commit này!')}>Xóa</button>
                  <button style={{ border: 'none', background: '#10b981', color: 'white', borderRadius: 6, padding: '0 10px', cursor: 'pointer' }} onClick={() => window.alert('Đánh dấu commit!')}>Đánh dấu</button>
                  <button style={{ border: 'none', background: '#6366f1', color: 'white', borderRadius: 6, padding: '0 10px', cursor: 'pointer' }} onClick={() => handleShowDetail(item)}>Chi tiết</button>
                </div>
              </Card>
            </List.Item>
          )}
          pagination={{
            current: commitPage,
            pageSize: commitPageSize,
            total: filtered.length,
            onChange: (page) => setCommitPage(page),
            showSizeChanger: false,
          }}
        />
        <CommitDetailModal
          visible={detailModalOpen}
          onClose={() => setDetailModalOpen(false)}
          commit={detailCommit}
          files={detailFiles}
          diffs={detailDiffs}
        />
      </>
    );
  };

  return (
    <Card
      title={<Title level={4}>🩺 Chuẩn đoán tổng hợp kho lưu trữ</Title>}
      style={{ marginBottom: 24 }}
      styles={{ body: { padding: 24 } }}
    >
      {/* Repo search and select box */}
      <div style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center' }}>
        <Input
          placeholder="Tìm kiếm repository..."
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          style={{ width: 220 }}
          disabled={loading}
        />
        <Select
          showSearch
          style={{ minWidth: 260 }}
          placeholder={loading ? 'Đang tải danh sách...' : 'Chọn repository'}
          value={repoId}
          onChange={handleRepoSelect}
          filterOption={false}
          loading={loading}
        >
          {filteredRepos.map(repo => (
            <Option key={repo.id} value={repo.id}>
              <Avatar src={repo.owner?.avatar_url} size={20} style={{ marginRight: 6 }} />
              <Tag color="blue">{repo.owner?.login}</Tag> / <Text strong>{repo.name}</Text>
            </Option>
          ))}
        </Select>
      </div>
      <Divider />
      {loading && (
        <div style={{ width: '100%', marginBottom: 16, textAlign: 'center' }}>
          <Spin />
          <div style={{ marginTop: 8, color: '#666' }}>Đang tải dữ liệu phân tích...</div>
        </div>
      )}
      {error && <Empty description={error} />}
      {!loading && !error && repoId && (
        <>
          <FadeInWrapper delay={0.1}>
            <Card title={<Text strong>Phân tích loại commit</Text>} size="small" style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(59,130,246,0.08)' }}>
              {commitAnalysis && commitAnalysis.statistics && commitAnalysis.statistics.commit_types ? (
                <CommitTypeChart
                  commitTypes={commitAnalysis.statistics.commit_types}
                  totalCommits={commitAnalysis.summary?.total_commits || 0}
                />
              ) : <Empty description="Không có dữ liệu commit." />}
              {/* Hiển thị danh sách commit nếu có */}
              {commitAnalysis && commitAnalysis.commits && commitAnalysis.commits.length > 0 && (
                <FadeInWrapper delay={0.2}>
                  <div style={{ marginTop: 24 }}>
                    <Text strong style={{ fontSize: 16 }}>Danh sách commit:</Text>
                    {renderCommitList(commitAnalysis.commits)}
                  </div>
                </FadeInWrapper>
              )}
              {/* Nếu không có, thử lấy từ allCommits */}
              {(!commitAnalysis || !commitAnalysis.commits || commitAnalysis.commits.length === 0) && allCommits && allCommits.commits && allCommits.commits.length > 0 && (
                <FadeInWrapper delay={0.2}>
                  <div style={{ marginTop: 24 }}>
                    <Text strong style={{ fontSize: 16 }}>Danh sách commit:</Text>
                    {renderCommitList(allCommits.commits)}
                  </div>
                </FadeInWrapper>
              )}
            </Card>
          </FadeInWrapper>
          <FadeInWrapper delay={0.3}>
            <Card title={<Text strong>Phân tích lĩnh vực công nghệ</Text>} size="small" style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(168,85,247,0.08)' }}>
              {areaAnalysis && areaAnalysis.areas ? (
                <List
                  dataSource={Object.entries(areaAnalysis.areas)}
                  renderItem={([area, value]) => (
                    <List.Item>
                      <Tag color="purple" style={{ fontSize: 15 }}>{area}</Tag>: <Text>{value}</Text>
                    </List.Item>
                  )}
                  style={{ paddingLeft: 0 }}
                />
              ) : <Empty description="Không có dữ liệu lĩnh vực." />}
            </Card>
          </FadeInWrapper>
          <FadeInWrapper delay={0.4}>
            <Card title={<Text strong>Phân tích rủi ro</Text>} size="small" style={{ borderRadius: 16, boxShadow: '0 2px 8px rgba(239,68,68,0.08)' }}>
              {riskAnalysis && riskAnalysis.risks ? (
                <List
                  dataSource={Object.entries(riskAnalysis.risks)}
                  renderItem={([risk, value]) => (
                    <List.Item>
                      <Tag color="red" style={{ fontSize: 15 }}>{risk}</Tag>: <Text>{value}</Text>
                    </List.Item>
                  )}
                  style={{ paddingLeft: 0 }}
                />
              ) : <Empty description="Không có dữ liệu rủi ro." />}
            </Card>
          </FadeInWrapper>
        </>
      )}
    </Card>
  );
};

export default RepoDiagnosisPanel;
