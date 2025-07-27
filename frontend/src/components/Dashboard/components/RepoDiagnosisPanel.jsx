  const [allMembersAnalysis, setAllMembersAnalysis] = useState(null);
  // Fetch all members commit analysis when repoId, repoSource, selectedBranch thay đổi
  useEffect(() => {
    if (!repoId || repoSource !== 'database') {
      setAllMembersAnalysis(null);
      return;
    }
    setLoading(true);
    const branchParam = selectedBranch ? `?branch_name=${encodeURIComponent(selectedBranch)}` : '';
    fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/members/commits-analyst${branchParam}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        setAllMembersAnalysis(data && data.data ? data.data : []);
      })
      .catch(() => {
        setAllMembersAnalysis([]);
      })
      .finally(() => {
        setLoading(false);
  const [allMembersAnalysis, setAllMembersAnalysis] = useState(null);

  // Fetch all members commit analysis when repoId, repoSource, selectedBranch thay đổi
  useEffect(() => {
    if (!repoId || repoSource !== 'database') {
      setAllMembersAnalysis(null);
      return;
    }
    setLoading(true);
    const branchParam = selectedBranch ? `?branch_name=${encodeURIComponent(selectedBranch)}` : '';
    fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/members/commits-analyst${branchParam}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        setAllMembersAnalysis(data && data.data ? data.data : []);
      })
      .catch(() => {
        setAllMembersAnalysis([]);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [repoId, repoSource, selectedBranch]);
      });
  }, [repoId, repoSource, selectedBranch]);
      {/* Hiển thị phân tích commit cho toàn bộ thành viên */}
      {allMembersAnalysis && allMembersAnalysis.length > 0 && (
        <FadeInWrapper delay={0.15}>
          <Card title={<Text strong>Phân tích commit toàn bộ thành viên</Text>} size="small" style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(59,130,246,0.08)' }}>
            {allMembersAnalysis.map(member => (
              <div key={member.member_login} style={{ marginBottom: 24 }}>
                <Text strong style={{ fontSize: 15 }}>
                  <Tag color="blue">{member.member_login}</Tag> - Tổng commit: {member.total_commits}
                </Text>
                {member.commits && member.commits.length > 0 ? (
                  <div style={{ marginTop: 8 }}>
                    {renderCommitList(member.commits)}
                  </div>
                ) : (
                  <Empty description="Không có commit cho thành viên này." />
                )}
                {member.statistics && member.statistics.commit_types && (
                  <div style={{ marginTop: 8 }}>
                    <CommitTypeChart commitTypes={member.statistics.commit_types} totalCommits={member.total_commits} />
                  </div>
                )}
                <Divider />
              </div>
            ))}
          </Card>
        </FadeInWrapper>
      )}
import React, { useEffect, useState } from 'react';
import { Card, Typography, Spin, Empty, Tag, Divider, List, Select, Input, Avatar, Button, Switch } from 'antd';
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
  const [repoSource, setRepoSource] = useState('database'); // 'database' | 'github'
  const [githubRepos, setGithubRepos] = useState([]);
  const [githubLoading, setGithubLoading] = useState(false);
  const [branchList, setBranchList] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState('');

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
      setCommitAnalysis(null);
      setAreaAnalysis(null);
      setRiskAnalysis(null);
      setAllCommits(null);
      return;
    }
    setLoading(true);
    setError(null);
    if (repoSource === 'github') {
      // Fetch commit analysis from GitHub API endpoint theo branch
      const token = localStorage.getItem('access_token');
      const repo = githubRepos.find(r => (r.id || r.github_id) === repoId);
      const owner = repo?.owner?.login || repo?.owner;
      const name = repo?.name;
      if (!owner || !name || !selectedBranch) {
        setLoading(false);
        return;
      }
      fetch(`http://localhost:8000/api/github/${owner}/${name}/branches/${selectedBranch}/commits?per_page=100&page=1`, {
        headers: { Authorization: `token ${token}` },
      })
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          setCommitAnalysis({ commits: data.commits || [], summary: { total_commits: data.count || 0 }, statistics: {} });
          setAreaAnalysis(null);
          setRiskAnalysis(null);
          setAllCommits(null);
        })
        .catch(() => {
          setError('Lỗi khi tải dữ liệu commit từ GitHub API.');
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      // DB source: truyền branch_name cho API /commits/all và /commits/all/analysis
      const branchParam = selectedBranch ? `?branch_name=${encodeURIComponent(selectedBranch)}` : '';
      Promise.all([
        fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all/analysis${branchParam}`).then(r => r.ok ? r.json() : null),
        fetch(`http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all${branchParam}`).then(r => r.ok ? r.json() : null),
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
    }
  }, [repoId, repoSource, githubRepos, selectedBranch]);

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
        {/* Branch select */}
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
