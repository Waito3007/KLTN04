import React, { useState, useEffect, useCallback } from 'react';
import { Card, Row, Col, Typography, Empty, message, Spin } from 'antd';

// Import các component con
import MemberList from './components/MemberList';
import CommitAnalyst from './components/CommitAnalyst';
import AreaAnalyst from './components/AreaAnalyst';
import RiskAnalyst from './components/RiskAnalyst'; // New import for RiskAnalyst
import CommitList from './components/CommitList';
import { Tabs } from 'antd';
import MultiFusionInsights from './components/MultiFusionInsights';
import ControlPanel from './components/ControlPanel';
import AIFeaturesPanel from './components/AIFeaturesPanel';
import OverviewStats from './components/OverviewStats';

const { Title } = Typography;

const RepositoryMembers = ({ selectedRepo }) => {  
  const [members, setMembers] = useState([]);
  const [selectedMember, setSelectedMember] = useState(null);
  const [memberCommits, setMemberCommits] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [showAIFeatures, setShowAIFeatures] = useState(false);
  const [useAI, setUseAI] = useState(true); // Toggle for AI analysis
  const [aiModel, setAiModel] = useState('han'); // 'han', 'multifusion', or 'multifusion-v2'
  const [aiModelStatus, setAiModelStatus] = useState(null);
  const [multiFusionV2Status, setMultiFusionV2Status] = useState(null); // MultiFusion V2 status
  const [branches, setBranches] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState(undefined); // Sửa: undefined thay vì null
  const [branchesLoading, setBranchesLoading] = useState(false);
  // State for commit filter, pagination, tab, multi-member mode
  const [commitTypeFilter, setCommitTypeFilter] = useState('all');
  const [techAreaFilter, setTechAreaFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 5;
  const [activeTab, setActiveTab] = useState('commitType'); // commitType | techArea | commitList
  const [multiMemberMode, setMultiMemberMode] = useState(false);
  const [multiMemberAnalysis, setMultiMemberAnalysis] = useState(null);

  // NEW STATES FOR FULL REPO ANALYSIS
  const [fullAreaAnalysis, setFullAreaAnalysis] = useState(null);
  const [fullRiskAnalysis, setFullRiskAnalysis] = useState(null);
  const [fullAnalysisLoading, setFullAnalysisLoading] = useState(false);
  const [allRepoCommits, setAllRepoCommits] = useState(null); // NEW: State for all repo commits
  const [allRepoCommitAnalysis, setAllRepoCommitAnalysis] = useState(null); // NEW: State for all repo commit analysis


  // Debug: Log component render and props
  console.log('RepositoryMembers RENDER:', { 
    selectedRepo, 
    members: members.length,
    loading,
    hasSelectedRepo: !!selectedRepo,
    repoId: selectedRepo?.id 
  });

  // Load AI model status
  const _loadAIModelStatus = useCallback(async () => {
    if (!selectedRepo?.id) return;
    
    try {
      // Load HAN model status - Using POST method since that's how the backend expects it
      const response = await fetch(`http://localhost:8000/api/repositories/${selectedRepo.id}/ai/model-status`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('AI Model Status:', data); // Debug log
        setAiModelStatus(data);
      } else {
        console.error('AI Status Error:', response.status);
      }

      // Load MultiFusion V2 status
      const multiFusionResponse = await fetch(`http://localhost:8000/api/repositories/${selectedRepo.id}/ai/model-v2-status`);
      
      if (multiFusionResponse.ok) {
        const multiFusionData = await multiFusionResponse.json();
        console.log('MultiFusion V2 Status:', multiFusionData); // Debug log
        setMultiFusionV2Status(multiFusionData);
      } else {
        console.error('MultiFusion V2 Status Error:', multiFusionResponse.status);
      }
    } catch (error) {
      console.error('Error loading AI model status:', error);
    }
  }, [selectedRepo?.id]);

  const loadRepositoryBranches = useCallback(async () => {
    if (!selectedRepo?.id) return;
    setBranchesLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/repositories/${selectedRepo.id}/branches`);
      if (response.ok) {
        const data = await response.json();
        setBranches(data.branches || []);
        if (typeof selectedBranch === 'undefined') {
          setSelectedBranch(undefined); // Sửa: undefined thay vì null
        }
      } else {
        console.error('Branches API Error:', response.status);
      }
    } catch (error) {
      console.error('Error loading branches:', error);
    } finally {
      setBranchesLoading(false);
    }
  }, [selectedRepo?.id, selectedBranch]);

  const loadRepositoryMembers = useCallback(async () => {
    if (!selectedRepo?.id) {
      console.log('❌ loadRepositoryMembers: No selectedRepo.id');
      return;
    }

    console.log('loadRepositoryMembers called with repo:', selectedRepo); // Debug log
    setLoading(true);
    try {
      const url = `http://localhost:8000/api/repositories/${selectedRepo.id}/members`;
      console.log('Fetching members from URL:', url); // Debug log
      
      // Test without token first
      const response = await fetch(url);
      
      console.log('Members API Response status:', response.status); // Debug log
      
      if (response.ok) {
        const data = await response.json();
        console.log('Members API Response data:', data); // Debug log
        setMembers(data.members || []);
        console.log('Members set:', data.members || []); // Debug log
      } else {
        console.error('Members API Error:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('Error response body:', errorText);
        message.error(`Không thể tải danh sách thành viên: ${response.status}`);
      }
    } catch (error) {
      console.error('Error loading members:', error);
      message.error('Lỗi khi tải thành viên');
    } finally {
      setLoading(false);
    }
  }, [selectedRepo]);

  // NEW: Load all repo commits
  const loadAllRepoCommits = useCallback(async () => {
    if (!selectedRepo?.id) return;
    setFullAnalysisLoading(true);
    try {
      const url = `http://localhost:8000/api/repositories/${selectedRepo.id}/commits/all`;
      console.log('Fetching all repo commits from URL:', url);
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        console.log('All Repo Commits Response:', data); // Debug log
        setAllRepoCommits(data.commits);
      } else {
        console.error('All Repo Commits API Error:', response.status, response.statusText, await response.text()); // Added await response.text()
        message.error(`Không thể tải tất cả commits của repo: ${response.status}`);
      }
    } catch (error) {
      console.error('Error loading all repo commits:', error);
      message.error('Lỗi khi tải tất cả commits của repo');
    } finally {
      setFullAnalysisLoading(false);
    }
  }, [selectedRepo?.id]);

  // NEW: Load all repo commit analysis
  const loadAllRepoCommitAnalysis = useCallback(async () => {
    if (!selectedRepo?.id) return;
    setFullAnalysisLoading(true);
    try {
      const url = `http://localhost:8000/api/repositories/${selectedRepo.id}/commits/all/analysis`;
      console.log('Fetching all repo commit analysis from URL:', url);
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        console.log('All Repo Commit Analysis Response:', data); // Debug log
        setAllRepoCommitAnalysis(data.analysis);
      } else {
        console.error('All Repo Commit Analysis API Error:', response.status, response.statusText, await response.text()); // Added await response.text()
        message.error(`Không thể tải phân tích commit toàn bộ repo: ${response.status}`);
      }
    } catch (error) {
      console.error('Error loading all repo commit analysis:', error);
      message.error('Lỗi khi tải phân tích commit toàn bộ repo');
    } finally {
      setFullAnalysisLoading(false);
    }
  }, [selectedRepo?.id]);

  // NEW: Load full area analysis for the repository
  const loadFullAreaAnalysis = useCallback(async () => {
    if (!selectedRepo?.id) return;
    setFullAnalysisLoading(true);
    try {
      const url = `http://localhost:8000/api/area-analysis/repositories/${selectedRepo.id}/full-area-analysis`;
      console.log('Fetching full area analysis from URL:', url);
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        console.log('Full Area Analysis Response:', data);
        setFullAreaAnalysis(data);
      } else {
        console.error('Full Area Analysis API Error:', response.status, response.statusText, await response.text()); // Added await response.text()
        message.error(`Không thể tải phân tích khu vực toàn diện: ${response.status}`);
      }
    } catch (error) {
      console.error('Error loading full area analysis:', error);
      message.error('Lỗi khi tải phân tích khu vực toàn diện');
    } finally {
      setFullAnalysisLoading(false);
    }
  }, [selectedRepo?.id]);

  // NEW: Load full risk analysis for the repository
  const loadFullRiskAnalysis = useCallback(async () => {
    if (!selectedRepo?.id) return;
    setFullAnalysisLoading(true); // Use the same loading state for now
    try {
      const url = `http://localhost:8000/api/risk-analysis/repositories/${selectedRepo.id}/full-risk-analysis`;
      console.log('Fetching full risk analysis from URL:', url);
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        console.log('Full Risk Analysis Response:', data);
        setFullRiskAnalysis(data);
      } else {
        console.error('Full Risk Analysis API Error:', response.status, response.statusText, await response.text()); // Added await response.text()
        message.error(`Không thể tải phân tích rủi ro toàn diện: ${response.status}`);
      }
    } catch (error) {
      console.error('Error loading full risk analysis:', error);
      message.error('Lỗi khi tải phân tích rủi ro toàn diện');
    } finally {
      setFullAnalysisLoading(false);
    }
  }, [selectedRepo?.id]);


  // Load members when repo changes
  useEffect(() => {
    console.log('RepositoryMembers useEffect triggered:', {
      selectedRepo,
      repoId: selectedRepo?.id,
      repoName: selectedRepo?.name,
      hasRepo: !!selectedRepo
    });
      if (selectedRepo && selectedRepo.id) {
      console.log('✅ Loading members for repo:', selectedRepo.name, 'ID:', selectedRepo.id);
      loadRepositoryMembers();
      loadRepositoryBranches();
      _loadAIModelStatus();
      // NEW: Trigger full analysis when repo changes
      loadFullAreaAnalysis();
      loadFullRiskAnalysis();
      loadAllRepoCommits(); // NEW
      loadAllRepoCommitAnalysis(); // NEW
    } else {
      console.log('❌ No selectedRepo or selectedRepo.id found:', {
        selectedRepo: !!selectedRepo,
        id: selectedRepo?.id
      });
      // Clear members if no repo
      setMembers([]);
      // Clear full analysis results
      setFullAreaAnalysis(null);
      setFullRiskAnalysis(null);
      setAllRepoCommits(null); // NEW
      setAllRepoCommitAnalysis(null); // NEW
    }  }, [selectedRepo, loadRepositoryMembers, loadRepositoryBranches, _loadAIModelStatus, loadFullAreaAnalysis, loadFullRiskAnalysis, loadAllRepoCommits, loadAllRepoCommitAnalysis]); // Add new dependencies

  // Re-analyze when branch or model changes
  useEffect(() => {
    if (selectedMember && (selectedBranch !== null || aiModel)) {
      console.log('Branch or model changed, re-analyzing member:', selectedMember.login, 'on branch:', selectedBranch, 'with model:', aiModel);
      handleMemberClick(selectedMember);
    }
  }, [selectedBranch, aiModel]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleMemberClick = async (member) => {
    setSelectedMember(member);
    setAnalysisLoading(true);
    
    try {
      let url;
      let requestData = null;

      if (aiModel === 'han' && useAI) {
        // Use original HAN analysis endpoint
        const aiParam = useAI ? '?use_ai=true' : '?use_ai=false';
        const branchParam = selectedBranch ? `&branch_name=${encodeURIComponent(selectedBranch)}` : '';
        url = `http://localhost:8000/api/repositories/${selectedRepo.id}/members/${member.login}/commits-han${aiParam}${branchParam}`;
      } else if (aiModel === 'multifusion' && useAI) {
        // Use MultiFusion V2 analysis endpoint
        url = `http://localhost:8000/api/repositories/${selectedRepo.id}/members/${member.login}/commits-v2`;
        const branchParam = selectedBranch ? `?branch_name=${encodeURIComponent(selectedBranch)}` : '';
        url += branchParam;
      } else {
        // Use original HAN analysis endpoint
        const aiParam = useAI ? '?use_ai=true' : '?use_ai=false';
        const branchParam = selectedBranch ? `&branch_name=${encodeURIComponent(selectedBranch)}` : '';
        url = `http://localhost:8000/api/repositories/${selectedRepo.id}/members/${member.login}/commits${aiParam}${branchParam}`;
      }

      console.log('Analysis URL:', url, 'Model:', aiModel); // Debug log
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        ...(requestData && { body: JSON.stringify(requestData) })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Commit Analysis Response:', data); // Debug log
        console.log('Response keys:', Object.keys(data)); // Debug keys
        console.log('AI Model used:', aiModel); // Debug AI model
        
        // Handle different response formats
        if (aiModel === 'han' && data.data) {
            setMemberCommits(data.data);
        } else if (aiModel === 'multifusion' && data.analysis) {
          // MultiFusion V2 response format
          console.log('Using MultiFusion response format');
          setMemberCommits({
            statistics: {
              commit_types: data.analysis.commit_type_distribution || {},
              tech_analysis: data.statistics?.tech_analysis || {},
              risk_analysis: data.statistics?.risk_analysis || {},
              productivity: data.analysis.productivity_metrics || {}
            },
            commits: data.commits || [],
            summary: {
              total_commits: data.analysis.total_commits || 0,
              ai_powered: true,
              model_used: 'MultiFusion V2',
              analysis_date: new Date().toISOString(),
              dominant_type: data.analysis.dominant_commit_type
            },
            multifusion_insights: data.analysis // Store full MultiFusion analysis
          });
        } else if (data.success && data.statistics && data.commits) {
          // HAN API response format (new format with success flag)
          console.log('Using HAN API response format');
          setMemberCommits({
            statistics: data.statistics,
            commits: data.commits,
            summary: {
              ai_powered: true,
              model_used: 'HAN AI',
              analysis_date: new Date().toISOString(),
              total_commits: data.commits.length
            },
          });
        } else if (data.statistics && data.commits) {
          // Direct HAN model response format  
          console.log('Using direct HAN response format');
          setMemberCommits({
            statistics: data.statistics,
            commits: data.commits,
            summary: data.summary || {
              ai_powered: true,
              model_used: 'HAN AI',
              analysis_date: new Date().toISOString()
            },
          });
        } else if (data.success && data.data) {
          // Alternative response format with data wrapper
          console.log('Using wrapped data format');
          setMemberCommits(data.data);
        } else if (data.data) {
          // Legacy format
          console.log('Using legacy data format');
          setMemberCommits(data.data);
        } else {
          setMemberCommits(null);
        }
      } else {
        console.error('Commit Analysis Error:', response.status, response.statusText);
        message.error(`Không thể phân tích commits: ${response.status}`);
      }
    } catch (error) {
      console.error('Error analyzing member:', error);
      message.error('Lỗi khi phân tích commits');
    } finally {
      setAnalysisLoading(false);
    }
  };

  if (!selectedRepo) {
    return (
      <Card>
        <Empty description="Vui lòng chọn repository để xem thành viên" />
      </Card>
    );
  }

  return (
    <div style={{ padding: '20px' }}>      {/* Header với AI Button */}      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'flex-start',
        marginBottom: '20px',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <Title level={3} style={{ margin: 0 }}>
          👥 Thành viên - {selectedRepo.name}
        </Title>
        
        <ControlPanel 
          branches={branches}
          selectedBranch={selectedBranch}
          setSelectedBranch={setSelectedBranch}
          branchesLoading={branchesLoading}
          aiModel={aiModel}
          setAiModel={setAiModel}
          useAI={useAI}
          setUseAI={setUseAI}
          aiModelStatus={aiModelStatus}
          multiFusionV2Status={multiFusionV2Status}
          showAIFeatures={showAIFeatures}
          setShowAIFeatures={setShowAIFeatures}
          // NEW PROPS FOR FULL ANALYSIS
          fullAnalysisLoading={fullAnalysisLoading}
          onAnalyzeFullRepo={() => {
            loadFullAreaAnalysis();
            loadFullRiskAnalysis();
            loadAllRepoCommits(); // NEW
            loadAllRepoCommitAnalysis(); // NEW
          }}
        />
      </div>

      {/* Thống kê tổng quan */}
      <OverviewStats members={members} branches={branches} />

      {/* AI Features Panel */}
      {showAIFeatures && (
        <AIFeaturesPanel
          aiModelStatus={aiModelStatus}
          multiFusionV2Status={multiFusionV2Status}
          useAI={useAI}
          aiModel={aiModel}
        />
      )}

      {fullAnalysisLoading && <Spin tip="Đang phân tích toàn bộ kho lưu trữ..." style={{ width: '100%', textAlign: 'center', marginBottom: '20px' }} />}

      <Row gutter={[24, 24]}>
        {/* MemberList luôn hiển thị bên trái để chọn thành viên */}
        <Col xs={24} md={8}>
          <MemberList 
            members={members} 
            loading={loading} 
            selectedMember={selectedMember} 
            onMemberClick={handleMemberClick}
          />
        </Col>
        {/* Member Analysis với Tabs bên phải */}
        <Col xs={24} md={16}>
          {(!multiMemberMode && !multiMemberAnalysis) && (
            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              items={[
                {
                  key: 'commitType',
                  label: '🏷️ Loại Commit',
                  children: (
                    <CommitAnalyst 
                      memberCommits={memberCommits} 
                      allRepoCommitAnalysis={allRepoCommitAnalysis} // NEW
                    />
                  )
                },
                {
                  key: 'techArea',
                  label: '🌐 Lĩnh vực công nghệ',
                  children: (
                    <AreaAnalyst 
                      memberCommits={memberCommits} 
                      fullAreaAnalysis={fullAreaAnalysis} // Pass new prop
                    />
                  )
                },
                {
                  key: 'riskAnalysis',
                  label: '⚠️ Phân tích rủi ro',
                  children: (
                    <RiskAnalyst 
                      memberCommits={memberCommits} 
                      fullRiskAnalysis={fullRiskAnalysis} // Pass new prop
                    />
                  )
                },
                {
                  key: 'commitList',
                  label: '📝 Danh sách commit gần đây',
                  children: (
                    <CommitList
                      memberCommits={memberCommits}
                      selectedMember={selectedMember}
                      selectedBranch={selectedBranch}
                      commitTypeFilter={commitTypeFilter}
                      setCommitTypeFilter={setCommitTypeFilter}
                      techAreaFilter={techAreaFilter}
                      setTechAreaFilter={setTechAreaFilter}
                      currentPage={currentPage}
                      setCurrentPage={setCurrentPage}
                      pageSize={pageSize}
                      allRepoCommits={allRepoCommits} // NEW
                    />
                  )
                }
              ]}
            />
          )}
        </Col>
      </Row>

      {/* MultiFusion V2 Insights Panel */}
      {memberCommits && memberCommits.multifusion_insights && (
        <MultiFusionInsights multifusionInsights={memberCommits.multifusion_insights} />
      )}
    </div>
  );
};

export default RepositoryMembers;
