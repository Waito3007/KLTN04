import React, { useEffect, useState } from 'react';
import { Card, Typography, Spin, Empty, Tag, Divider, Select, Input, Avatar, Button, Switch } from 'antd';
import BranchCommitAnalysis from '../CommitAnalyst/BranchCommitAnalysis';
import CommitList from '../CommitAnalyst/CommitList';
import AreaAnalysis from '../AreaAnalyst/AreaAnalysis';
import RiskAnalysis from '../RiskAnalyst/RiskAnalysis';

const { Title, Text } = Typography;
const { Option } = Select;

const RepoDiagnosisPanel = ({ repositories = [], onRepoChange, onBranchChange }) => {
  const [selectedMemberArea, setSelectedMemberArea] = useState('');
  const [compareAreaMode, setCompareAreaMode] = useState(false);
  const [compareMemberArea, setCompareMemberArea] = useState('');
  const [selectedMemberRisk, setSelectedMemberRisk] = useState('');
  const [compareRiskMode, setCompareRiskMode] = useState(false);
  const [compareMemberRisk, setCompareMemberRisk] = useState('');
  const [selectedMember, setSelectedMember] = useState('');
  const [error, setError] = useState(null);
  const [areaAnalysis, setAreaAnalysis] = useState(null);
  const [riskAnalysis, setRiskAnalysis] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [repoId, setRepoId] = useState(null);
  const [repoSource, setRepoSource] = useState('database');
  const [githubRepos, setGithubRepos] = useState([]);
  const [githubLoading, setGithubLoading] = useState(false);
  const [branchList, setBranchList] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [branchAnalysis, setBranchAnalysis] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  const filteredRepos = (repoSource === 'github' ? githubRepos : repositories).filter(
    repo =>
      repo.name.toLowerCase().includes(searchText.toLowerCase()) ||
      (repo.owner?.login || repo.owner || '').toLowerCase().includes(searchText.toLowerCase())
  );

  useEffect(() => {
    if (repoSource === 'database' && repositories.length > 0 && !repoId) {
      setRepoId(repositories[0].id);
      if (onRepoChange)
        onRepoChange(repositories[0]);
    }
    if (repoSource === 'github' && githubRepos.length > 0 && !repoId) {
      setRepoId(githubRepos[0].id || githubRepos[0].github_id || githubRepos[0].id);
      if (onRepoChange)
        onRepoChange(githubRepos[0]);
    }
  }, [repositories, githubRepos, repoSource, repoId, onRepoChange]);

  useEffect(() => {
    setBranchList([]); // Clear branchList immediately on repoId change
    setSelectedBranch(''); // Clear selectedBranch immediately on repoId change

    if (!repoId) {
      return;
    }
    const fetchBranches = async () => {
      if (repoSource === 'github') {
        const repo = githubRepos.find(r => (r.id || r.github_id) === repoId);
        if (!repo)
          return;
        const token = localStorage.getItem('access_token');
        const owner = repo.owner?.login || repo.owner;
        const name = repo.name;
        const res = await fetch(`http://localhost:8000/api/github/${owner}/${name}/branches`, {
          headers: { Authorization: `token ${token}` },
        });
        const data = await res.json();
        setBranchList(Array.isArray(data) ? data : []);
        if (data && data.length > 0 && data[0].name) // Ensure data[0].name exists
          setSelectedBranch(data[0].name);
        else {
          setSelectedBranch(''); // Clear if no branches found
        }
      } else {
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
        if (data.branches && data.branches.length > 0 && data.branches[0].name) // Ensure data.branches[0].name exists
          setSelectedBranch(data.branches[0].name);
        else {
          setSelectedBranch(''); // Clear if no branches found
        }
      }
    };
    fetchBranches();
  }, [repoId, repoSource, githubRepos, repositories]);

  const handleAnalyze = async () => {
    if (!repoId || !selectedBranch)
      return;

    setAnalysisLoading(true);
    setError(null);
    setBranchAnalysis(null);
    setAreaAnalysis(null);
    setRiskAnalysis(null);

    try {
      const commitAnalysisUrl = `http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all/analysis?branch_name=${encodeURIComponent(selectedBranch)}`;
      const areaAnalysisUrl = `http://localhost:8000/api/area-analysis/repositories/${repoId}/full-area-analysis?branch_name=${encodeURIComponent(selectedBranch)}`;
      const riskAnalysisUrl = `http://localhost:8000/api/risk-analysis/repositories/${repoId}/full-risk-analysis?branch_name=${encodeURIComponent(selectedBranch)}`;

      const [commitRes, areaRes, riskRes] = await Promise.all([
        fetch(commitAnalysisUrl),
        fetch(areaAnalysisUrl),
        fetch(riskAnalysisUrl),
      ]);

      if (!commitRes.ok)
        throw new Error('Lỗi khi phân tích commit cho nhánh này.');
      if (!areaRes.ok)
        throw new Error('Lỗi khi tải dữ liệu phân tích lĩnh vực.');
      if (!riskRes.ok)
        throw new new Error('Lỗi khi tải dữ liệu phân tích rủi ro.');

      const commitData = await commitRes.json();
      const areaData = await areaRes.json();
      const riskData = await riskRes.json();

      setBranchAnalysis(commitData);
      setAreaAnalysis(areaData);
      setRiskAnalysis(riskData);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleRepoSelect = id => {
    setRepoId(id);
    if (onRepoChange) {
      const repo = repositories.find(r => r.id === id);
      onRepoChange(repo);
    }
  };

  const handleBranchChange = (branch) => {
    setSelectedBranch(branch);
    if (onBranchChange) {
      onBranchChange(branch);
    }
  };

  const handleSourceToggle = async checked => {
    if (checked) {
      setRepoSource('github');
      setGithubLoading(true);
      const token = localStorage.getItem('access_token');
      try {
        const res = await fetch('http://localhost:8000/api/github/repos', {
          headers: { Authorization: `token ${token}` },
        });
        const data = await res.json();
        setGithubRepos(Array.isArray(data) ? data : []);
        if (data && data.length > 0) {
          setRepoId(data[0].id || data[0].github_id || data[0].id);
          if (onRepoChange)
            onRepoChange(data[0]);
        }
      } catch {
        setGithubRepos([]);
      } finally {
        setGithubLoading(false);
      }
    } else {
      setRepoSource('database');
      setRepoId(repositories[0]?.id || null);
      if (onRepoChange && repositories[0])
        onRepoChange(repositories[0]);
    }
  };

  return (
    <Card
      title={<Title level={4}>🩺 Chuẩn đoán tổng hợp kho lưu trữ</Title>}
      style={{ marginBottom: 24 }}
      styles={{ body: { padding: 24 } }}
    >
      <div style={{ marginBottom: 16, display: 'flex', gap: 16, alignItems: 'center' }}>
        <Switch
          checked={repoSource === 'github'}
          onChange={handleSourceToggle}
          checkedChildren="GitHub API"
          unCheckedChildren="Database"
          disabled={githubLoading}
        />
        <span style={{ fontWeight: 500, color: repoSource === 'github' ? '#3b82f6' : '#64748b' }}>
          {repoSource === 'github' ? 'Đang lấy từ GitHub API' : 'Đang lấy từ Database'}
        </span>
        <Input
          placeholder="Tìm kiếm repository..."
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          style={{ width: 220 }}
          disabled={githubLoading}
        />
        <Select
          showSearch
          style={{ minWidth: 260 }}
          placeholder={githubLoading ? 'Đang tải danh sách...' : 'Chọn repository'}
          value={repoId}
          onChange={handleRepoSelect}
          filterOption={false}
          loading={githubLoading}
        >
          {filteredRepos.map(repo => (
            <Option key={repo.id || repo.github_id} value={repo.id || repo.github_id}>
              <Avatar src={repo.owner?.avatar_url || repo.owner_avatar_url} size={20} style={{ marginRight: 6 }} />
              <Tag color="blue">{repo.owner?.login || repo.owner}</Tag> / <Text strong>{repo.name}</Text>
            </Option>
          ))}
        </Select>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Select
            style={{ minWidth: 180 }}
            placeholder={branchList.length === 0 ? 'Không có branch' : 'Chọn branch'}
            value={selectedBranch}
            onChange={handleBranchChange}
            disabled={githubLoading || branchList.length === 0}
          >
            {branchList.map(branch => (
              <Option key={branch.name || branch} value={branch.name || branch}>
                {branch.name || branch}
              </Option>
            ))}
          </Select>
          <Button type="primary" onClick={handleAnalyze} disabled={!selectedBranch || analysisLoading} loading={analysisLoading}>
            Phân tích
          </Button>
        </div>
      </div>
      <Divider />
      {analysisLoading && (
        <div style={{ width: '100%', marginBottom: 16, textAlign: 'center' }}>
          <Spin />
          <div style={{ marginTop: 8, color: '#666' }}>Đang tải dữ liệu phân tích...</div>
        </div>
      )}
      {error && <Empty description={error} />}
      {!analysisLoading && !error && repoId && (
        <>
          {branchAnalysis && (
            <BranchCommitAnalysis
              branchAnalysis={branchAnalysis}
              selectedMember={selectedMember}
              setSelectedMember={setSelectedMember}
              renderCommitList={commits => <CommitList commits={commits} />}
            />
          )}
          {areaAnalysis && (
            <AreaAnalysis
              areaAnalysis={areaAnalysis}
              selectedMemberArea={selectedMemberArea}
              setSelectedMemberArea={setSelectedMemberArea}
              compareAreaMode={compareAreaMode}
              setCompareAreaMode={setCompareAreaMode}
              compareMemberArea={compareMemberArea}
              setCompareMemberArea={setCompareMemberArea}
            />
          )}
          {riskAnalysis && (
            <RiskAnalysis
              riskAnalysis={riskAnalysis}
              selectedMemberRisk={selectedMemberRisk}
              setSelectedMemberRisk={setSelectedMemberRisk}
              compareRiskMode={compareRiskMode}
              setCompareRiskMode={setCompareRiskMode}
              compareMemberRisk={compareMemberRisk}
              setCompareMemberRisk={compareMemberRisk}
            />
          )}
        </>
      )}
    </Card>
  );
};

export default RepoDiagnosisPanel;
