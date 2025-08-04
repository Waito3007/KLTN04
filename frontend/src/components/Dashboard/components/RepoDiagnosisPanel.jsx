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
  const [compareMemberRisk] = useState('');
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
    console.log('🔄 useEffect triggered - repoId changed:', repoId);
    setBranchList([]); // Clear branchList immediately on repoId change
    setSelectedBranch(''); // Clear selectedBranch immediately on repoId change

    if (!repoId) {
      console.log('❌ No repoId, skipping branch fetch');
      return;
    }
    const fetchBranches = async () => {
      console.log('🔍 Fetching branches for repoId:', repoId, 'source:', repoSource);
      
      if (repoSource === 'github') {
        const repo = githubRepos.find(r => (r.id || r.github_id) === repoId);
        if (!repo) {
          console.log('❌ GitHub repo not found for id:', repoId);
          return;
        }
        const token = localStorage.getItem('access_token');
        const owner = repo.owner?.login || repo.owner;
        const name = repo.name;
        console.log('🌐 GitHub API call:', { owner, name });
        
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
        if (!repo) {
          setBranchList([]);
          setSelectedBranch('');
          return;
        }
        
        // Handle cả trường hợp owner là string hoặc object
        const owner = typeof repo.owner === 'string' 
          ? repo.owner 
          : repo.owner?.login || repo.owner?.name || repo.owner;
        const name = repo.name;
        
        console.log('🔍 Fetching branches for:', { owner, name, repo });
        
        if (!owner || !name) {
          console.error('❌ Missing owner or name:', { owner, name });
          setBranchList([]);
          setSelectedBranch('');
          return;
        }
        
        try {
          const res = await fetch(`http://localhost:8000/api/commits/${owner}/${name}/branches`);
          const data = await res.json();
          console.log('✅ Branch data received:', data);
          
          setBranchList(Array.isArray(data.branches) ? data.branches : []);
          if (data.branches && data.branches.length > 0 && data.branches[0].name) {
            setSelectedBranch(data.branches[0].name);
          } else {
            setSelectedBranch('');
          }
        } catch (error) {
          console.error('❌ Error fetching branches:', error);
          setBranchList([]);
          setSelectedBranch('');
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
      const commitAnalysisUrl = `http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all/analysis?branch_name=${encodeURIComponent(selectedBranch)}&limit=1000`;
      const areaAnalysisUrl = `http://localhost:8000/api/area-analysis/repositories/${repoId}/full-area-analysis?branch_name=${encodeURIComponent(selectedBranch)}&limit_per_member=1000`;
      const riskAnalysisUrl = `http://localhost:8000/api/risk-analysis/repositories/${repoId}/full-risk-analysis?branch_name=${encodeURIComponent(selectedBranch)}&limit_per_member=1000`;

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
    console.log('🔄 Repository selected:', id);
    setRepoId(id);
    const repo = (repoSource === 'github' ? githubRepos : repositories).find(r => (r.id || r.github_id) === id);
    console.log('🔍 Found repository:', repo);
    if (onRepoChange && repo) {
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
    // Reset các state khi chuyển nguồn dữ liệu
    setRepoId(null);
    setSelectedBranch('');
    setBranchList([]);
    setBranchAnalysis(null);
    setAreaAnalysis(null);
    setRiskAnalysis(null);
    
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
        // Không tự động chọn repo đầu tiên - để user tự chọn
      } catch {
        setGithubRepos([]);
      } finally {
        setGithubLoading(false);
      }
    } else {
      setRepoSource('database');
      // Không tự động chọn repo đầu tiên - để user tự chọn
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
          {filteredRepos.map(repo => {
            // Handle owner có thể là string hoặc object
            const ownerName = typeof repo.owner === 'string' 
              ? repo.owner 
              : repo.owner?.login || repo.owner?.name || repo.owner;
            const avatarUrl = typeof repo.owner === 'object' 
              ? repo.owner?.avatar_url 
              : repo.owner_avatar_url;
            
            return (
              <Option key={repo.id || repo.github_id} value={repo.id || repo.github_id}>
                <Avatar src={avatarUrl} size={20} style={{ marginRight: 6 }} />
                <Tag color="blue">{ownerName}</Tag> / <Text strong>{repo.name}</Text>
              </Option>
            );
          })}
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
              areaLoading={false}
              selectedMemberArea={selectedMemberArea}
              setSelectedMemberArea={setSelectedMemberArea}
              compareAreaMode={compareAreaMode}
              setCompareAreaMode={setCompareAreaMode}
              compareMemberArea={compareMemberArea}
              setCompareMemberArea={setCompareMemberArea}
              selectedBranch={selectedBranch}
            />
          )}
          {riskAnalysis && (
            <RiskAnalysis
              riskAnalysis={riskAnalysis}
              riskLoading={false}
              selectedMemberRisk={selectedMemberRisk}
              setSelectedMemberRisk={setSelectedMemberRisk}
              compareRiskMode={compareRiskMode}
              setCompareRiskMode={setCompareRiskMode}
              compareMemberRisk={compareMemberRisk}
              setCompareMemberRisk={compareMemberRisk}
              selectedBranch={selectedBranch}
            />
          )}
        </>
      )}
    </Card>
  );
};

export default RepoDiagnosisPanel;
