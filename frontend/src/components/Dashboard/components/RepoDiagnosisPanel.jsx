import React, { useEffect, useState } from 'react';
import { Card, Typography, Spin, Empty, Tag, Divider, List, Select, Input, Avatar, Button, Switch } from 'antd';
import { Pie } from 'react-chartjs-2';
import FadeInWrapper from './FadeInWrapper';
import CommitDetailModal from './CommitDetailModal';

const { Title, Text } = Typography;
const { Option } = Select;

// Modern, unified diagnosis panel for all repo analyses
const RepoDiagnosisPanel = ({ repositories = [], onRepoChange }) => {
  // State cho lọc thành viên
  const [selectedMember, setSelectedMember] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [commitAnalysis, setCommitAnalysis] = useState(null);
  const [areaAnalysis, setAreaAnalysis] = useState(null);
  const [riskAnalysis, setRiskAnalysis] = useState(null);
  const [allCommits, setAllCommits] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [repoId, setRepoId] = useState(null);
  const [repoSource, setRepoSource] = useState('database'); // 'database' | 'github'
  const [githubRepos, setGithubRepos] = useState([]);
  const [githubLoading, setGithubLoading] = useState(false);
  const [branchList, setBranchList] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [branchAnalysis, setBranchAnalysis] = useState(null);
  const [branchAnalysisLoading, setBranchAnalysisLoading] = useState(false);
  const [branchAnalysisError, setBranchAnalysisError] = useState(null);

  // Filtered repo list for search
  const filteredRepos = (repoSource === 'github' ? githubRepos : repositories).filter(repo =>
    repo.name.toLowerCase().includes(searchText.toLowerCase()) ||
    (repo.owner?.login || repo.owner || '').toLowerCase().includes(searchText.toLowerCase())
  );

  useEffect(() => {
    // Auto-select first repo if available
    if (repoSource === 'database' && repositories.length > 0 && !repoId) {
      setRepoId(repositories[0].id);
      if (onRepoChange) onRepoChange(repositories[0]);
    }
    if (repoSource === 'github' && githubRepos.length > 0 && !repoId) {
      setRepoId(githubRepos[0].id || githubRepos[0].github_id || githubRepos[0].id);
      if (onRepoChange) onRepoChange(githubRepos[0]);
    }
  }, [repositories, githubRepos, repoSource]);

  // Fetch branch list when repoId or repoSource changes
  useEffect(() => {
    if (!repoId) {
      setBranchList([]);
      setSelectedBranch('');
      return;
    }
    const fetchBranches = async () => {
      if (repoSource === 'github') {
        const repo = githubRepos.find(r => (r.id || r.github_id) === repoId);
        if (!repo) return;
        const token = localStorage.getItem('access_token');
        const owner = repo.owner?.login || repo.owner;
        const name = repo.name;
        const res = await fetch(`http://localhost:8000/api/github/${owner}/${name}/branches`, {
          headers: { Authorization: `token ${token}` },
        });
        const data = await res.json();
        setBranchList(Array.isArray(data) ? data : []);
        if (data && data.length > 0) setSelectedBranch(data[0].name);
      } else {
        // DB source: lấy owner và name từ repositories
        const repo = repositories.find(r => r.id === repoId);
        const owner = repo?.owner?.login || repo?.owner;
        const name = repo?.name;
        if (!owner || !name) {
          setBranchList([]);
          setSelectedBranch('');
          return;
        }
        const res = await fetch(`http://localhost:8000/api/commits/${owner}/${name}/branches`);
        const data = await res.json();
        setBranchList(Array.isArray(data.branches) ? data.branches : []);
        if (data.branches && data.branches.length > 0) setSelectedBranch(data.branches[0].name);
      }
    };
    fetchBranches();
  }, [repoId, repoSource, githubRepos]);

  // Fetch analysis data when repoId changes
  useEffect(() => {
    if (!repoId) {
      setAreaAnalysis(null);
      setRiskAnalysis(null);
      setAllCommits(null);
      return;
    }
    setLoading(true);
    setError(null);
    // Chỉ fetch các phân tích tổng thể, không fetch commit analysis theo branch tự động
    Promise.all([
      fetch(`http://localhost:8000/api/area-analysis/repositories/${repoId}/full-area-analysis`).then(r => r.ok ? r.json() : null),
      fetch(`http://localhost:8000/api/risk-analysis/repositories/${repoId}/full-risk-analysis`).then(r => r.ok ? r.json() : null),
      fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all`).then(r => r.ok ? r.json() : null)
    ]).then(([areaData, riskData, allData]) => {
      setAreaAnalysis(areaData);
      setRiskAnalysis(riskData);
      setAllCommits(allData);
    }).catch(() => {
      setError('Lỗi khi tải dữ liệu phân tích kho lưu trữ.');
    }).finally(() => {
      setLoading(false);
    });
  }, [repoId, repoSource, githubRepos]);

  // Hàm thực hiện phân tích commit theo branch khi ấn nút
  const handleAnalyzeBranch = async () => {
    if (!repoId || !selectedBranch) return;
    setBranchAnalysisLoading(true);
    setBranchAnalysisError(null);
    let url = `http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all/analysis?branch_name=${encodeURIComponent(selectedBranch)}`;
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error('Lỗi khi phân tích commit cho nhánh này.');
      const data = await resp.json();
      setBranchAnalysis(data);
    } catch (err) {
      setBranchAnalysisError(err.message);
      setBranchAnalysis(null);
    } finally {
      setBranchAnalysisLoading(false);
    }
  };

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
    // Nếu là github, lấy files và diff từ commit.files
    if (repoSource === 'github' && commit.files) {
      setDetailFiles(commit.files.map(f => ({
        filename: f.filename,
        status: f.status,
        language: '', // Có thể dùng thư viện detect language nếu cần
      })));
      const diffs = {};
      commit.files.forEach(f => {
        diffs[f.filename] = f.patch || '';
      });
      setDetailDiffs(diffs);
    } else {
      // Demo/fallback cho database
      setDetailFiles(commit.files || [
        { filename: 'src/App.js', status: 'modified', language: 'javascript' },
        { filename: 'README.md', status: 'added', language: 'markdown' }
      ]);
      setDetailDiffs({
        'src/App.js': `-console.log('Hello')\n+console.log('Hello World!')`,
        'README.md': `+# Project\n+This is a new README file.`
      });
    }
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
                  {item.branch_name && <Tag color="magenta">{item.branch_name}</Tag>}
                  {item.analysis && (
                    <>
                      {item.analysis.type && <Tag color="geekblue">{item.analysis.type}</Tag>}
                      {item.analysis.ai_powered && <Tag color="green">AI</Tag>}
                      {typeof item.analysis.confidence === 'number' && <Tag color="gold">Độ tự tin: {Math.round(item.analysis.confidence * 100)}%</Tag>}
                      {item.analysis.ai_model && <Tag color="purple">{item.analysis.ai_model}</Tag>}
                    </>
                  )}
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

  // Handler for switching repo source
  const handleSourceToggle = async (checked) => {
    if (checked) {
      setRepoSource('github');
      setGithubLoading(true);
      // Fetch from GitHub API
      const token = localStorage.getItem('access_token');
      try {
        const res = await fetch('http://localhost:8000/api/github/repos', {
          headers: { Authorization: `token ${token}` },
        });
        const data = await res.json();
        setGithubRepos(Array.isArray(data) ? data : []);
        // Auto-select first repo from GitHub
        if (data && data.length > 0) {
          setRepoId(data[0].id || data[0].github_id || data[0].id);
          if (onRepoChange) onRepoChange(data[0]);
        }
      } catch {
        setGithubRepos([]);
      } finally {
        setGithubLoading(false);
      }
    } else {
      setRepoSource('database');
      setRepoId(repositories[0]?.id || null);
      if (onRepoChange && repositories[0]) onRepoChange(repositories[0]);
    }
  };

  return (
    <Card
      title={<Title level={4}>🩺 Chuẩn đoán tổng hợp kho lưu trữ</Title>}
      style={{ marginBottom: 24 }}
      styles={{ body: { padding: 24 } }}
    >
      {/* Repo source toggle */}
      <div style={{ marginBottom: 16, display: 'flex', gap: 16, alignItems: 'center' }}>
        <Switch
          checked={repoSource === 'github'}
          onChange={handleSourceToggle}
          checkedChildren="GitHub API"
          unCheckedChildren="Database"
          disabled={loading || githubLoading}
        />
        <span style={{ fontWeight: 500, color: repoSource === 'github' ? '#3b82f6' : '#64748b' }}>
          {repoSource === 'github' ? 'Đang lấy từ GitHub API' : 'Đang lấy từ Database'}
        </span>
        <Input
          placeholder="Tìm kiếm repository..."
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          style={{ width: 220 }}
          disabled={loading || githubLoading}
        />
        <Select
          showSearch
          style={{ minWidth: 260 }}
          placeholder={loading || githubLoading ? 'Đang tải danh sách...' : 'Chọn repository'}
          value={repoId}
          onChange={handleRepoSelect}
          filterOption={false}
          loading={loading || githubLoading}
        >
          {filteredRepos.map(repo => (
            <Option key={repo.id || repo.github_id} value={repo.id || repo.github_id}>
              <Avatar src={repo.owner?.avatar_url || repo.owner_avatar_url} size={20} style={{ marginRight: 6 }} />
              <Tag color="blue">{repo.owner?.login || repo.owner}</Tag> / <Text strong>{repo.name}</Text>
            </Option>
          ))}
        </Select>
        {/* Branch select + nút phân tích */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Select
            style={{ minWidth: 180 }}
            placeholder={branchList.length === 0 ? 'Không có branch' : 'Chọn branch'}
            value={selectedBranch}
            onChange={setSelectedBranch}
            disabled={loading || githubLoading || branchList.length === 0}
          >
            {branchList.map(branch => (
              <Option key={branch.name || branch} value={branch.name || branch}>{branch.name || branch}</Option>
            ))}
          </Select>
          <Button
            type="primary"
            onClick={handleAnalyzeBranch}
            disabled={!selectedBranch || branchAnalysisLoading}
            loading={branchAnalysisLoading}
          >Phân tích</Button>
        </div>
      </div>
      <Divider />
      {(loading || githubLoading) && (
        <div style={{ width: '100%', marginBottom: 16, textAlign: 'center' }}>
          <Spin />
          <div style={{ marginTop: 8, color: '#666' }}>Đang tải dữ liệu phân tích...</div>
        </div>
      )}
      {error && <Empty description={error} />}
      {!loading && !error && repoId && (
        <>
          <FadeInWrapper delay={0.1}>
            <Card title={<Text strong>Phân tích loại commit theo nhánh</Text>} size="small" style={{ marginBottom: 32, borderRadius: 28, boxShadow: '0 4px 24px rgba(59,130,246,0.12)', background: '#f6f8fc', border: '1px solid #e0e7ef' }}>
              {branchAnalysisLoading && <Spin />}
              {branchAnalysisError && <Empty description={branchAnalysisError} />}
              <div style={{ width: '100%', background: '#fff', borderRadius: 24, boxShadow: '0 2px 12px rgba(59,130,246,0.08)', padding: 24, display: 'flex', flexDirection: 'row', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
                {/* Table left, chart right */}
                <div style={{ flex: 1, minWidth: 320 }}>
                  <Text strong style={{ fontSize: 16, marginBottom: 16, display: 'block' }}>Lọc theo thành viên:</Text>
                  <Select
                    style={{ minWidth: 180, marginBottom: 16 }}
                    placeholder="Chọn thành viên"
                    value={selectedMember}
                    onChange={setSelectedMember}
                    allowClear
                    disabled={!branchAnalysis || !branchAnalysis.commits}
                  >
                    <Select.Option value="">Tất cả</Select.Option>
                    {(branchAnalysis && branchAnalysis.commits)
                      ? Array.from(new Set(branchAnalysis.commits.map(c => c.author_name || 'Không rõ'))).map(author => (
                          <Select.Option key={author} value={author}>{author}</Select.Option>
                        ))
                      : null}
                  </Select>
                  <Text strong style={{ fontSize: 16, marginBottom: 8, display: 'block' }}>Loại commit / Số lượng:</Text>
                  <List
                    dataSource={(() => {
                      if (!branchAnalysis || !branchAnalysis.commits) return [];
                      let typeCount = {};
                      let filtered = selectedMember
                        ? branchAnalysis.commits.filter(c => (c.author_name || 'Không rõ') === selectedMember)
                        : branchAnalysis.commits;
                      filtered.forEach(c => {
                        const type = c.analysis?.type || 'other';
                        typeCount[type] = (typeCount[type] || 0) + 1;
                      });
                      return Object.entries(typeCount);
                    })()}
                    renderItem={([type, count]) => (
                      <List.Item>
                        <Tag color="blue" style={{ fontSize: 15 }}>{type}</Tag>: <Text>{count}</Text>
                      </List.Item>
                    )}
                    style={{ paddingLeft: 0 }}
                  />
                </div>
                <div style={{ flex: 1, minWidth: 320, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Text strong style={{ fontSize: 16, marginBottom: 16 }}>Biểu đồ loại commit:</Text>
                  {branchAnalysis && branchAnalysis.commits && (() => {
                    let typeCount = {};
                    let filtered = selectedMember
                      ? branchAnalysis.commits.filter(c => (c.author_name || 'Không rõ') === selectedMember)
                      : branchAnalysis.commits;
                    filtered.forEach(c => {
                      const type = c.analysis?.type || 'other';
                      typeCount[type] = (typeCount[type] || 0) + 1;
                    });
                    const pieLabels = Object.keys(typeCount);
                    const pieValues = Object.values(typeCount);
                    if (pieLabels.length === 0) return <Empty description="Không có dữ liệu." />;
                    return (
                      <div style={{ width: 320, height: 320 }}>
                        <Pie
                          data={{
                            labels: pieLabels,
                            datasets: [
                              {
                                data: pieValues,
                                backgroundColor: ["#8b5cf6", "#6366f1", "#3b82f6", "#06b6d4", "#f59e42", "#ef4444", "#22c55e"],
                              },
                            ],
                          }}
                          options={{
                            plugins: {
                              legend: { position: 'right', labels: { font: { size: 14 } } },
                            },
                          }}
                        />
                      </div>
                    );
                  })()}
                </div>
              </div>
              {/* Danh sách commit đã lọc */}
              <div style={{ marginTop: 32, padding: '0 24px' }}>
                <Text strong style={{ fontSize: 16 }}>Danh sách commit:</Text>
                {(branchAnalysis && branchAnalysis.commits)
                  ? (selectedMember
                      ? renderCommitList(branchAnalysis.commits.filter(c => (c.author_name || 'Không rõ') === selectedMember))
                      : renderCommitList(branchAnalysis.commits)
                    )
                  : <Empty description="Chưa có dữ liệu commit cho nhánh này." />
                }
              </div>
            </Card>
          </FadeInWrapper>
          <FadeInWrapper delay={0.3}>
            <Card title={<Text strong>Phân tích lĩnh vực công nghệ</Text>} size="small" style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(168,85,247,0.08)' }}>
              {areaAnalysis && areaAnalysis.area_distribution ? (
                <>
                  <div style={{ width: 320, height: 320, margin: '0 auto' }}>
                    <Pie
                      data={{
                        labels: Object.keys(areaAnalysis.area_distribution),
                        datasets: [
                          {
                            data: Object.values(areaAnalysis.area_distribution),
                            backgroundColor: ["#8b5cf6", "#6366f1", "#3b82f6", "#06b6d4", "#f59e42", "#ef4444", "#22c55e"],
                          },
                        ],
                      }}
                      options={{
                        plugins: {
                          legend: { position: 'right', labels: { font: { size: 14 } } },
                        },
                      }}
                    />
                  </div>
                  <List
                    dataSource={Object.entries(areaAnalysis.area_distribution)}
                    renderItem={([area, value]) => (
                      <List.Item>
                        <Tag color="purple" style={{ fontSize: 15 }}>{area}</Tag>: <Text>{value}</Text>
                      </List.Item>
                    )}
                    style={{ paddingLeft: 0 }}
                  />
                  <Divider />
                  <Title level={5}>Phân tích theo thành viên</Title>
                  <List
                    dataSource={areaAnalysis.members_area_analysis}
                    renderItem={member => (
                      <List.Item>
                        <Text strong>{member.member_login}</Text>
                        <List
                          dataSource={Object.entries(member.area_summary)}
                          renderItem={([area, value]) => (
                            <span style={{ marginRight: 12 }}>
                              <Tag color="blue">{area}</Tag>: <Text>{value}</Text>
                            </span>
                          )}
                          style={{ display: 'inline-block' }}
                        />
                      </List.Item>
                    )}
                  />
                </>
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