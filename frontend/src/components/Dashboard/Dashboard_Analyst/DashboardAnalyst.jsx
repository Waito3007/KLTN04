import React, { useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import { Card, Space, Typography, Tag, Select, Collapse, List, Progress, Row, Col, Statistic, Alert, Button, Drawer } from 'antd';
import styled from 'styled-components';
import axios from 'axios';
import { 
  WarningOutlined, InfoCircleOutlined, BranchesOutlined, LineChartOutlined, 
  ExperimentOutlined, TeamOutlined, UserSwitchOutlined, ClockCircleOutlined, ThunderboltOutlined,
  RightOutlined, ExpandAltOutlined, CloseOutlined, RobotOutlined, SyncOutlined
} from '@ant-design/icons';
import Widget from "@components/common/Widget";
import { Loading } from '@components/common';

const ExpandButton = styled(Button)`
  position: absolute;
  top: 12px;
  right: 12px;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  background: #ffffff;
  color: #64748b;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  
  &:hover {
    border-color: #3b82f6;
    color: #3b82f6;
    background: #f8fafc;
    transform: scale(1.05);
  }
  
  &:active {
    transform: scale(0.98);
  }
`;

const ScrollableContent = styled.div`
  max-height: calc(100vh - 300px);
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 8px;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
`;

const DrawerContent = styled.div`
  .ant-collapse-ghost > .ant-collapse-item > .ant-collapse-content { 
    padding: 12px 0; 
  }
  .ant-collapse-header { 
    padding: 16px 0 !important; 
    font-size: 16px !important;
    font-weight: 600 !important;
  }
`;

const AnalyzeButton = styled(Button)`
  width: 100%;
  height: 36px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 8px;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
  
  &:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  }
  
  &:active {
    transform: translateY(0);
  }
  
  &:disabled {
    background: #f1f5f9;
    color: #94a3b8;
    box-shadow: none;
    transform: none;
  }
`;

// Styled components
const SidebarCard = styled(Card)`
  position: relative;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  .ant-card-body { padding: 16px; }
  .ant-card-head { padding: 12px 16px; border-bottom: 1px solid #f1f5f9; }
  .ant-collapse-ghost > .ant-collapse-item > .ant-collapse-content { padding: 0; }
  .ant-collapse-header { padding: 8px 0 !important; }
`;

const SectionTitle = styled(Typography.Title)`
  margin-bottom: 0 !important;
  font-weight: 600 !important;
  color: #1e293b !important;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const StatisticItem = ({ title, value, suffix, precision = 1 }) => (
  <div>
    <Typography.Text type="secondary" style={{ fontSize: '12px', display: 'block' }}>{title}</Typography.Text>
    <Typography.Text strong style={{ fontSize: '16px' }}>
      {typeof value === 'number' ? value.toFixed(precision) : value}
      {suffix && <span style={{ fontSize: '12px', marginLeft: '4px', color: '#475569' }}>{suffix}</span>}
    </Typography.Text>
  </div>
);

// --- Analysis Sections ---

const ProgressSection = ({ data }) => {
  if (!data || data.total_commits === 0) return <Typography.Text type="secondary">Không có dữ liệu tiến độ.</Typography.Text>;
  const commitTypes = data.commits_by_type ? Object.entries(data.commits_by_type) : [];
  return (
    <Space direction="vertical" style={{ width: '100%' }} size="middle">
      <Row gutter={16}>
        <Col span={12}><StatisticItem title="Total Commits" value={data.total_commits} precision={0} /></Col>
        <Col span={12}><StatisticItem title="Velocity" value={data.velocity} suffix="c/day" /></Col>
      </Row>
      <div>
        <Typography.Text type="secondary" style={{ fontSize: '12px' }}>Productivity Score</Typography.Text>
        <Progress percent={data.productivity_score} strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }} />
      </div>
      {commitTypes.length > 0 && (
         <div>
          <Typography.Text strong style={{ fontSize: '12px' }}>Commit Types:</Typography.Text>
          <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {commitTypes.map(([type, count]) => <Tag key={type}>{type}: {count}</Tag>)}
          </div>
        </div>
      )}
      {data.recommendations?.length > 0 && (
        <div>
          <Typography.Text strong style={{ fontSize: '12px' }}>Gợi ý:</Typography.Text>
          <List size="small" dataSource={data.recommendations.slice(0,2)} renderItem={item => <List.Item style={{padding: '2px 0', fontSize: '12px', border: 'none' }}>- {item}</List.Item>} />
        </div>
      )}
    </Space>
  );
};

const RisksSection = ({ data }) => {
  if (!data || data.risk_score === 0) return <Typography.Text type="secondary">Không có dữ liệu rủi ro.</Typography.Text>;
  return (
    <Space direction="vertical" style={{ width: '100%' }} size="middle">
      <Row gutter={16}>
        <Col span={12}><StatisticItem title="Risk Score" value={data.risk_score} suffix="%" /></Col>
        <Col span={12}><StatisticItem title="High-Risk Commits" value={data.high_risk_commits?.length || 0} precision={0} /></Col>
      </Row>
      {data.critical_areas?.length > 0 && (
        <div>
          <Typography.Text strong style={{ fontSize: '12px' }}>Khu vực trọng yếu:</Typography.Text>
          <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {data.critical_areas.map(area => <Tag color="volcano" key={area}>{area}</Tag>)}
          </div>
        </div>
      )}
      {data.mitigation_suggestions?.length > 0 && (
        <div>
          <Typography.Text strong style={{ fontSize: '12px' }}>Gợi ý giảm thiểu:</Typography.Text>
          <List size="small" dataSource={data.mitigation_suggestions.slice(0,2)} renderItem={item => <List.Item style={{padding: '2px 0', fontSize: '12px', border: 'none' }}>- {item}</List.Item>} />
        </div>
      )}
    </Space>
  );
};

const ProductivitySection = ({ data, onDeveloperClick }) => {
  if (!data || !data.team_summary) return <Typography.Text type="secondary">Không có dữ liệu năng suất.</Typography.Text>;
  const summary = data.team_summary;
  const memberMetrics = data.member_metrics ? Object.entries(data.member_metrics).sort(([, a], [, b]) => b.commits - a.commits) : [];

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="middle">
      <Row gutter={16}>
        <Col span={12}><StatisticItem title="Lines Changed" value={summary.total_lines_changed} precision={0} /></Col>
        <Col span={12}><StatisticItem title="Fix Ratio" value={summary.fix_ratio} suffix="%" /></Col>
      </Row>
      <Row gutter={16}>
        <Col span={12}><StatisticItem title="Active Contributors" value={summary.active_contributors} precision={0} /></Col>
        <Col span={12}><StatisticItem title="Avg. Commit Size" value={summary.average_commit_size} precision={0} suffix="lines" /></Col>
      </Row>
      {memberMetrics.length > 0 && (
        <div>
          <Typography.Text strong style={{ fontSize: '12px' }}>Top Contributors (click to view DNA):</Typography.Text>
          <List
            size="small"
            dataSource={memberMetrics}
            renderItem={([name, metrics]) => (
              <List.Item 
                style={{padding: '2px 0', fontSize: '12px', border: 'none', cursor: 'pointer' }}
                onClick={() => onDeveloperClick(name)}
              >
                <Typography.Link>{name}: {metrics.commits} commits, {metrics.lines_added + metrics.lines_removed} lines</Typography.Link>
              </List.Item>
            )}
          />
        </div>
      )}
    </Space>
  );
};

const AssignmentSection = ({ data }) => {
  if (!data || data.length === 0) return <Typography.Text type="secondary">Không có gợi ý phân công.</Typography.Text>;
  const sorted = [...data].sort((a, b) => (b.skill_match_score - b.workload_score) - (a.skill_match_score - a.workload_score));
  return (
    <List
      size="small"
      dataSource={sorted.slice(0, 3)}
      renderItem={item => (
        <List.Item style={{padding: '4px 0', border: 'none'}}>
          <List.Item.Meta
            title={<span style={{fontSize: '13px'}}>{item.member_name}</span>}
            description={<span style={{fontSize: '12px'}}>{item.rationale}</span>}
          />
        </List.Item>
      )}
    />
  );
};

const DnaProfileDisplay = ({ dna, loading }) => {
  if (loading) return <Loading variant="circle" size="small" message="Analyzing DNA..." />;
  if (!dna) return <p>Select a developer to see their DNA profile.</p>;
  if (dna.message) return <Alert message={dna.message} type="info" />;

  const { work_rhythm, contribution_style, tech_expertise, risk_profile } = dna;

  const getPrimaryStyleTag = (style) => {
    const styles = {
      feature: { color: 'success', name: 'Builder' },
      fix: { color: 'error', name: 'Stabilizer' },
      refactor: { color: 'processing', name: 'Refiner' },
      docs: { color: 'default', name: 'Documenter' },
      test: { color: 'warning', name: 'Tester' },
    };
    return <Tag color={styles[style]?.color || 'default'}>{styles[style]?.name || style}</Tag>;
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="Work Pattern & Rhythm">
        <Statistic title="Primary Style" valueRender={() => getPrimaryStyleTag(contribution_style.primary_style)} />
        <Statistic title="Commit Frequency" value={work_rhythm.commit_frequency} suffix="commits/day" />
        <Statistic title="Avg. Commit Size" value={risk_profile.avg_commit_size} precision={0} suffix="lines" />
      </Card>
      <Card title="Contribution Style">
        {Object.entries(contribution_style.distribution).map(([type, count]) => (
          <div key={type} style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>{type}</span>
            <Tag>{count}</Tag>
          </div>
        ))}
      </Card>
      <Card title="Technical Expertise">
        <Typography.Text strong>Areas:</Typography.Text>
        <div style={{ margin: '8px 0' }}>
          {Object.keys(tech_expertise.areas).map(area => <Tag key={area} color="blue">{area}</Tag>)}
        </div>
        <Typography.Text strong>Languages:</Typography.Text>
        <div style={{ marginTop: '8px' }}>
          {Object.keys(tech_expertise.languages).map(lang => <Tag key={lang} color="geekblue">{lang}</Tag>)}
        </div>
      </Card>
      <Card title="Risk Profile">
        <Progress percent={risk_profile.high_risk_percentage} steps={5} strokeColor={['#52c41a', '#faad14', '#f5222d']} />
        <Statistic title="High Risk Commits" value={`${risk_profile.high_risk_percentage}%`} />
      </Card>
    </Space>
  );
};


// --- Main Component ---

import { useEffect } from 'react';

const DashboardAnalyst = forwardRef(({ selectedRepoId, repositories, onBranchChange, /* hideRepoSelector = false, */ fullWidth = false }, ref) => {
  // State declarations - PHẢI ĐẶT TRƯỚC TẤT CẢ HOOKS KHÁC
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [branches, setBranches] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [branchLoading, setBranchLoading] = useState(false);
  const [daysBack, setDaysBack] = useState(30);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [dnaDrawerOpen, setDnaDrawerOpen] = useState(false);
  const [selectedDeveloper, setSelectedDeveloper] = useState(null);
  const [dnaLoading, setDnaLoading] = useState(false);
  const [dnaData, setDnaData] = useState(null);

  // Tự động load danh sách nhánh khi selectedRepoId thay đổi
  const fetchBranches = async () => {
    if (!selectedRepoId) {
      setBranches([]);
      setSelectedBranch('');
      return;
    }
    const token = localStorage.getItem('access_token');
    if (!token) return;
    const selectedRepo = repositories.find(repo => repo.id === selectedRepoId);
    if (!selectedRepo) return;
    const ownerName = typeof selectedRepo.owner === 'string' ? selectedRepo.owner : selectedRepo.owner?.login;
    
    setBranchLoading(true);
    try {
      const response = await axios.get(`http://localhost:8000/api/commits/${ownerName}/${selectedRepo.name}/branches`, { headers: { Authorization: `Bearer ${token}` } });
      const branchList = (response.data.branches || []).map(b => ({ value: b.name, label: b.name, isDefault: b.is_default || b.name === 'main' || b.name === 'master' }));
      setBranches(branchList);
      const defaultBranch = branchList.find(b => b.isDefault) || branchList[0];
      if (defaultBranch) {
        setSelectedBranch(defaultBranch.value);
        if (onBranchChange) onBranchChange(defaultBranch.value);
      }
    } catch (err) {
      console.error("Lỗi khi tải branches:", err);
      setBranches([]);
    } finally {
      setBranchLoading(false);
    }
  };

  useEffect(() => {
    if (selectedRepoId) {
      fetchBranches();
    } else {
      setBranches([]);
      setSelectedBranch('');
      setAnalyticsData(null); // Clear analytics data when no repo selected
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRepoId]);

  // Fetch comprehensive analytics - CHỈ GỌI KHI USER CHỌN REPO VÀ NHẤN NÚT
  const fetchAnalytics = useCallback(async () => {
    if (!selectedRepoId) {
      setAnalyticsData(null);
      return;
    }

    const token = localStorage.getItem('access_token');
    if (!token) {
      setError("Yêu cầu xác thực.");
      return;
    }

    const selectedRepo = repositories.find(repo => repo.id === selectedRepoId);
    if (!selectedRepo) {
      setError("Repository không tồn tại.");
      return;
    }

    // FIX: Prioritize owner's name over login for the analytics API endpoint
    const ownerName = typeof selectedRepo.owner === 'string' ? selectedRepo.owner : selectedRepo.owner?.name || selectedRepo.owner?.login;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(
        `http://localhost:8000/api/dashboard/analytics/${ownerName}/${selectedRepo.name}`,
        {
          headers: { Authorization: `Bearer ${token}` },
          params: { days_back: daysBack }
        }
      );
      setAnalyticsData(response.data);
      console.log('✅ DashboardAnalyst: Comprehensive analytics loaded:', response.data);
    } catch (err) {
      console.error("Lỗi khi tải phân tích dashboard:", err);
      setError("Không thể tải dữ liệu phân tích. Vui lòng thử lại.");
      setAnalyticsData(null);
    } finally {
      setLoading(false);
    }
  }, [selectedRepoId, repositories, daysBack]);

  // Expose fetchAnalytics method via ref
  useImperativeHandle(ref, () => ({
    fetchAnalytics
  }), [fetchAnalytics]);

  const handleDeveloperClick = async (authorName, forceRefresh = false) => {
    setSelectedDeveloper(authorName);
    setDnaDrawerOpen(true);
    setDnaLoading(true);

    const token = localStorage.getItem('access_token');
    const selectedRepo = repositories.find(repo => repo.id === selectedRepoId);
    if (!token || !selectedRepo) return;

    const ownerName = typeof selectedRepo.owner === 'string' ? selectedRepo.owner : selectedRepo.owner?.name || selectedRepo.owner?.login;

    try {
      const response = await axios.get(
        `http://localhost:8000/api/dashboard/dna/${ownerName}/${selectedRepo.name}/${authorName}`,
        {
          headers: { Authorization: `Bearer ${token}` },
          params: { 
            days_back: 90, // Use a longer period for DNA analysis
            force_refresh: forceRefresh
          }
        }
      );
      setDnaData(response.data);
    } catch (err) {
      console.error("Lỗi khi tải DNA developer:", err);
      setDnaData({ message: `Không thể tải dữ liệu cho ${authorName}.` });
    } finally {
      setDnaLoading(false);
    }
  };

  const handleBranchChange = (branchName) => {
    setSelectedBranch(branchName);
    if (onBranchChange) {
      onBranchChange(branchName);
    }
  };

  const renderContent = () => {
    if (loading) return <div style={{textAlign: 'center', padding: '20px'}}><Loading variant="circle" size="large" message="Đang phân tích dữ liệu..." /></div>;
    if (error) return <Alert message={error} type="error" showIcon />;
    if (!selectedRepoId) return (
      <div style={{textAlign: 'center', padding: '20px', color: '#64748b'}}>
        <LineChartOutlined style={{fontSize: '24px', marginBottom: '8px'}} />
        <Typography.Text type="secondary" style={{display: 'block'}}>Chọn repository để bắt đầu phân tích</Typography.Text>
      </div>
    );
    if (!analyticsData) return (
      <div style={{textAlign: 'center', padding: '20px', color: '#64748b'}}>
        <ThunderboltOutlined style={{fontSize: '24px', marginBottom: '8px'}} />
        <Typography.Text type="secondary" style={{display: 'block'}}>Nhấn "Phân tích AI" để xem kết quả phân tích</Typography.Text>
      </div>
    );

    return (
      <ScrollableContent>
        <Collapse 
          defaultActiveKey={['progress', 'risks']} 
          ghost
          items={[
            {
              key: 'progress',
              label: <><LineChartOutlined /> Tiến độ</>,
              children: <ProgressSection data={analyticsData.progress} />
            },
            {
              key: 'risks',
              label: <><ExperimentOutlined /> Rủi ro</>,
              children: <RisksSection data={analyticsData.risks} />
            },
            {
              key: 'productivity',
              label: <><TeamOutlined /> Năng suất</>,
              children: <ProductivitySection data={analyticsData.productivity_metrics} onDeveloperClick={handleDeveloperClick} />
            },
            {
              key: 'assignment',
              label: <><UserSwitchOutlined /> Gợi ý phân công</>,
              children: <AssignmentSection data={analyticsData.assignment_suggestions} />
            }
          ]}
        />
      </ScrollableContent>
    );
  };

  return (
    <>
      {fullWidth ? (
        // Full-width layout for dedicated page
        <div style={{ width: '100%' }}>
          {!selectedRepoId ? (
            <div style={{textAlign: 'center', padding: '60px 20px', color: '#64748b'}}>
              <LineChartOutlined style={{fontSize: '48px', marginBottom: '16px'}} />
              <Typography.Title level={4} type="secondary">Chọn repository để bắt đầu phân tích</Typography.Title>
              <Typography.Text type="secondary">Sử dụng dropdown ở phía trên để chọn repository cần phân tích</Typography.Text>
            </div>
          ) : !analyticsData ? (
            <div style={{textAlign: 'center', padding: '60px 20px', color: '#64748b'}}>
              <ThunderboltOutlined style={{fontSize: '48px', marginBottom: '16px'}} />
              <Typography.Title level={4} type="secondary">Sẵn sàng phân tích AI</Typography.Title>
              <Typography.Text type="secondary">Nhấn nút "Chạy phân tích AI" để bắt đầu phân tích repository</Typography.Text>
            </div>
          ) : (
            <div style={{ width: '100%' }}>
              <Collapse 
                defaultActiveKey={['progress', 'risks', 'productivity', 'assignment']} 
                size="large"
                items={[
                  {
                    key: 'progress',
                    label: (
                      <div style={{ fontSize: '16px', fontWeight: 500 }}>
                        <LineChartOutlined /> Tiến độ dự án
                      </div>
                    ),
                    children: <ProgressSection data={analyticsData.progress} />
                  },
                  {
                    key: 'risks',
                    label: (
                      <div style={{ fontSize: '16px', fontWeight: 500 }}>
                        <ExperimentOutlined /> Phân tích rủi ro
                      </div>
                    ),
                    children: <RisksSection data={analyticsData.risks} />
                  },
                  {
                    key: 'productivity',
                    label: (
                      <div style={{ fontSize: '16px', fontWeight: 500 }}>
                        <TeamOutlined /> Năng suất thành viên
                      </div>
                    ),
                    children: <ProductivitySection data={analyticsData.productivity_metrics} onDeveloperClick={handleDeveloperClick} />
                  },
                  {
                    key: 'assignment',
                    label: (
                      <div style={{ fontSize: '16px', fontWeight: 500 }}>
                        <UserSwitchOutlined /> Gợi ý phân công
                      </div>
                    ),
                    children: <AssignmentSection data={analyticsData.assignment_suggestions} />
                  }
                ]}
              />
            </div>
          )}
        </div>
      ) : (
        // Original sidebar widget layout
        <SidebarCard
          title={
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <SectionTitle level={5} style={{ fontSize: '14px', margin: 0 }}>Bảng phân tích AI</SectionTitle>
              <ExpandButton 
                onClick={() => setDrawerOpen(true)}
                icon={<ExpandAltOutlined />}
                title="Mở rộng để xem đầy đủ"
                size="small"
              />
            </div>
          }
          size="small"
        >
          {selectedRepoId && (
            <Space direction="vertical" style={{width: '100%', marginBottom: '12px'}}>
              <div>
                <Typography.Text type="secondary" style={{ fontSize: '12px', marginBottom: '4px', display: 'block' }}><BranchesOutlined /> Nhánh:</Typography.Text>
                <Select
                  style={{ width: '100%' }}
                  placeholder="Chọn nhánh"
                  value={selectedBranch}
                  onChange={handleBranchChange}
                  loading={branchLoading}
                  size="small"
                  disabled={!branches.length}
                >
                  {branches.map(branch => (
                    <Select.Option key={branch.value} value={branch.value}>
                      {branch.label}
                      {branch.isDefault && <Tag size="small" color="blue" style={{ marginLeft: 4 }}>default</Tag>}
                    </Select.Option>
                  ))}
                </Select>
              </div>
              <div>
                <Typography.Text type="secondary" style={{ fontSize: '12px', marginBottom: '4px', display: 'block' }}><ClockCircleOutlined /> Khoảng thời gian:</Typography.Text>
                <Select value={daysBack} onChange={setDaysBack} style={{ width: '100%' }} size="small">
                  <Select.Option value={7}>7 ngày qua</Select.Option>
                  <Select.Option value={30}>30 ngày qua</Select.Option>
                  <Select.Option value={90}>90 ngày qua</Select.Option>
                </Select>
              </div>
              <AnalyzeButton 
                type="primary" 
                icon={<ThunderboltOutlined />}
                onClick={fetchAnalytics}
                loading={loading}
                disabled={!selectedRepoId || !selectedBranch}
              >
                {loading ? 'Đang phân tích...' : 'Phân tích AI'}
              </AnalyzeButton>
            </Space>
          )}
          
          {renderContent()}
        </SidebarCard>
      )}

      <Drawer
        title={
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <RobotOutlined style={{ marginRight: '8px' }} />
            AI Phân tích - Xem đầy đủ
          </div>
        }
        placement="right"
        width={600}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        closeIcon={<CloseOutlined />}
      >
        <DrawerContent>
          {analyticsData ? (
            <Collapse 
              defaultActiveKey={['progress', 'risks', 'productivity', 'assignment']} 
              ghost
              items={[
                {
                  key: 'progress',
                  label: <><LineChartOutlined /> Tiến độ</>,
                  children: <ProgressSection data={analyticsData.progress} />
                },
                {
                  key: 'risks',
                  label: <><ExperimentOutlined /> Rủi ro</>,
                  children: <RisksSection data={analyticsData.risks} />
                },
                {
                  key: 'productivity',
                  label: <><TeamOutlined /> Năng suất</>,
                  children: <ProductivitySection data={analyticsData.productivity_metrics} onDeveloperClick={handleDeveloperClick} />
                },
                {
                  key: 'assignment',
                  label: <><UserSwitchOutlined /> Gợi ý phân công</>,
                  children: <AssignmentSection data={analyticsData.assignment_suggestions} />
                }
              ]}
            />
          ) : (
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
              <RobotOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
              <p>Chưa có dữ liệu phân tích</p>
              <p>Vui lòng chọn repository và nhấn "Phân tích AI"</p>
            </div>
          )}
        </DrawerContent>
      </Drawer>

      <Drawer
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>{`Developer DNA: ${selectedDeveloper}`}</span>
            <Button 
              icon={<SyncOutlined />} 
              onClick={() => handleDeveloperClick(selectedDeveloper, true)}
              loading={dnaLoading}
            >
              Refresh
            </Button>
          </div>
        }
        placement="right"
        width={400}
        open={dnaDrawerOpen}
        onClose={() => setDnaDrawerOpen(false)}
      >
        <DnaProfileDisplay dna={dnaData} loading={dnaLoading} />
      </Drawer>
    </>
  );
});

export default DashboardAnalyst;