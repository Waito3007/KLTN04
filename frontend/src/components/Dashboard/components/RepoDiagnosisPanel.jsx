import React, { useEffect, useState } from 'react';
import { Typography, Empty, Tag, Divider, Select, Input, Avatar, Button, Switch, Tabs } from 'antd';
import Card from '@components/common/Card';
import { Loading } from '@components/common';
import BranchCommitAnalysis from "@components/Dashboard/CommitAnalyst/BranchCommitAnalysis";
import CommitList from "@components/Dashboard/CommitAnalyst/CommitList";
import AreaAnalysis from "@components/Dashboard/AreaAnalyst/AreaAnalysis";
import RiskAnalysis from "@components/Dashboard/RiskAnalyst/RiskAnalysis";

const { Title, Text } = Typography;
const { Option } = Select;

const RepoDiagnosisPanel = ({ repositories = [], selectedRepoId, onRepoChange, onBranchChange, hideRepoSelector = false }) => {
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
  const [activeTabKey, setActiveTabKey] = useState('1'); // Quản lý tab đang hoạt động

  const filteredRepos = (repoSource === 'github' ? githubRepos : repositories).filter(
    repo =>
      repo.name.toLowerCase().includes(searchText.toLowerCase()) ||
      (repo.owner?.login || repo.owner || '').toLowerCase().includes(searchText.toLowerCase())
  );
  // Sử dụng selectedRepoId từ props khi hideRepoSelector = true
  const effectiveRepoId = hideRepoSelector ? selectedRepoId : repoId;

  useEffect(() => {
    if (hideRepoSelector && selectedRepoId) {
      setRepoId(selectedRepoId);
    }
  }, [hideRepoSelector, selectedRepoId]);

  useEffect(() => {
    console.log('🔄 useEffect triggered - repoId changed:', effectiveRepoId);
    setBranchList([]); // Clear branchList immediately on repoId change
    setSelectedBranch(''); // Clear selectedBranch immediately on repoId change

    if (!effectiveRepoId) {
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
  }, [effectiveRepoId, repoId, repoSource, githubRepos, repositories]);

  const handleCommitAnalysis = async () => {
    if (!repoId || !selectedBranch) return;

    setAnalysisLoading(true);
    setError(null);
    setBranchAnalysis(null);

    try {
      const commitAnalysisUrl = `http://localhost:8000/api/multifusion-commit-analysis/${repoId}/commits/all/analysis?branch_name=${encodeURIComponent(selectedBranch)}&limit=1000`;
      const commitRes = await fetch(commitAnalysisUrl);

      if (!commitRes.ok) throw new Error('Lỗi khi phân tích commit cho nhánh này.');

      const commitData = await commitRes.json();
      setBranchAnalysis(commitData);
      setActiveTabKey('1'); // Chuyển sang tab Phân tích Commit
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleAreaAnalysis = async () => {
    if (!repoId || !selectedBranch) return;

    setAnalysisLoading(true);
    setError(null);
    setAreaAnalysis(null);

    try {
      const areaAnalysisUrl = `http://localhost:8000/api/area-analysis/repositories/${repoId}/full-area-analysis?branch_name=${encodeURIComponent(selectedBranch)}&limit_per_member=1000`;
      const areaRes = await fetch(areaAnalysisUrl);

      if (!areaRes.ok) throw new Error('Lỗi khi tải dữ liệu phân tích lĩnh vực.');

      const areaData = await areaRes.json();
      setAreaAnalysis(areaData);
      setActiveTabKey('2'); // Chuyển sang tab Phân tích Phạm vi
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalysisLoading(false);
    }
  };

  const handleRiskAnalysis = async () => {
    if (!repoId || !selectedBranch) return;

    setAnalysisLoading(true);
    setError(null);
    setRiskAnalysis(null);

    try {
      const riskAnalysisUrl = `http://localhost:8000/api/risk-analysis/repositories/${repoId}/full-risk-analysis?branch_name=${encodeURIComponent(selectedBranch)}&limit_per_member=1000`;
      const riskRes = await fetch(riskAnalysisUrl);

      if (!riskRes.ok) throw new Error('Lỗi khi tải dữ liệu phân tích rủi ro.');

      const riskData = await riskRes.json();
      setRiskAnalysis(riskData);
      setActiveTabKey('3'); // Chuyển sang tab Phân tích Rủi ro
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
      {!hideRepoSelector && (
        <>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 16,
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
              <Switch
                checked={repoSource === 'github'}
                onChange={handleSourceToggle}
                checkedChildren="GitHub API"
                unCheckedChildren="Database"
                disabled={githubLoading}
              />
              <span
                style={{
                  fontWeight: 500,
                  color: repoSource === 'github' ? '#3b82f6' : '#64748b',
                }}
              >
                {repoSource === 'github' ? 'Đang lấy từ GitHub API' : 'Đang lấy từ Database'}
              </span>
              <Input
                placeholder="Tìm kiếm repository..."
                value={searchText}
                onChange={e => setSearchText(e.target.value)}
                style={{ width: '100%', maxWidth: 220 }}
                disabled={githubLoading}
              />
            </div>

            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
              <Select
                showSearch
                style={{ minWidth: 260, maxWidth: '100%' }}
                placeholder={githubLoading ? 'Đang tải danh sách...' : 'Chọn repository'}
                value={repoId}
                onChange={handleRepoSelect}
                filterOption={false}
                loading={githubLoading}
              >
                {filteredRepos.map(repo => {
                  const ownerName =
                    typeof repo.owner === 'string'
                      ? repo.owner
                      : repo.owner?.login || repo.owner?.name || repo.owner;
                  const avatarUrl =
                    typeof repo.owner === 'object'
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
            </div>
          </div>
          <Divider />
        </>
      )}
      
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 16,
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <Select
            style={{ minWidth: 180, maxWidth: '100%' }}
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
        </div>

        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button
            type="primary"
            onClick={handleCommitAnalysis}
            disabled={!selectedBranch || analysisLoading}
            loading={analysisLoading}
          >
            Phân tích Commit
          </Button>
          <Button
            type="primary"
            onClick={handleAreaAnalysis}
            disabled={!selectedBranch || analysisLoading}
            loading={analysisLoading}
          >
            Phân tích Phạm vi
          </Button>
          <Button
            type="primary"
            onClick={handleRiskAnalysis}
            disabled={!selectedBranch || analysisLoading}
            loading={analysisLoading}
          >
            Phân tích Rủi ro
          </Button>
        </div>
      </div>

      <Divider />

      {analysisLoading && (
        <div style={{ width: '100%', marginBottom: 16, textAlign: 'center' }}>
          <Loading variant="circle" size="small" message="Đang tải dữ liệu phân tích..." />
        </div>
      )}

      {error && <Empty description={error} />}

      {!analysisLoading && !error && repoId && (
        <Tabs 
          activeKey={activeTabKey} 
          onChange={setActiveTabKey}
          items={[
            {
              key: '1',
              label: 'Phân tích Commit',
              children: branchAnalysis ? (
                <BranchCommitAnalysis
                  branchAnalysis={branchAnalysis}
                  selectedMember={selectedMember}
                  setSelectedMember={setSelectedMember}
                  renderCommitList={commits => <CommitList commits={commits} />}
                />
              ) : (
                <Empty description="Chưa có dữ liệu phân tích Commit. Vui lòng thực hiện phân tích." />
              )
            },
            {
              key: '2', 
              label: 'Phân tích Phạm vi',
              children: areaAnalysis ? (
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
              ) : (
                <Empty description="Chưa có dữ liệu phân tích Phạm vi. Vui lòng thực hiện phân tích." />
              )
            },
            {
              key: '3',
              label: 'Phân tích Rủi ro', 
              children: riskAnalysis ? (
                <RiskAnalysis
                  riskAnalysis={riskAnalysis}
                  riskLoading={false}
                  selectedMemberRisk={selectedMemberRisk}
                  setSelectedMemberRisk={setSelectedMemberRisk}
                  compareRiskMode={compareRiskMode}
                  setCompareRiskMode={setCompareRiskMode}
                  compareMemberRisk={compareMemberRisk}
                  setCompareMemberRisk={setCompareMemberRisk}
                  selectedBranch={selectedBranch}
                />
              ) : (
                <Empty description="Chưa có dữ liệu phân tích Rủi ro. Vui lòng thực hiện phân tích." />
              )
            }
          ]}
        />
      )}
    </Card>
  );
};

export default RepoDiagnosisPanel;
