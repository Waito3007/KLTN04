import React, { useState } from 'react';
import { Card, Space, Typography, Spin, Tag, Select, Collapse, List, Progress, Row, Col, Statistic, Alert } from 'antd';
import styled from 'styled-components';
import axios from 'axios';
import { 
  WarningOutlined, InfoCircleOutlined, BranchesOutlined, LineChartOutlined, 
  ExperimentOutlined, TeamOutlined, UserSwitchOutlined, ClockCircleOutlined
} from '@ant-design/icons';

// Styled components
const SidebarCard = styled(Card)`
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

const ProductivitySection = ({ data }) => {
  if (!data || !data.team_summary) return <Typography.Text type="secondary">Không có dữ liệu năng suất.</Typography.Text>;
  const summary = data.team_summary;
  const memberMetrics = data.member_metrics ? Object.entries(data.member_metrics).sort(([, a], [, b]) => b.commits - a.commits).slice(0, 3) : [];

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
          <Typography.Text strong style={{ fontSize: '12px' }}>Top Contributors:</Typography.Text>
          <List
            size="small"
            dataSource={memberMetrics}
            renderItem={([name, metrics]) => (
              <List.Item style={{padding: '2px 0', fontSize: '12px', border: 'none' }}>
                {name}: {metrics.commits} commits, {metrics.lines_added + metrics.lines_removed} lines
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


// --- Main Component ---

import { useEffect } from 'react';

const DashboardAnalyst = ({ selectedRepoId, repositories, onBranchChange }) => {
  // Tự động load danh sách nhánh khi selectedRepoId thay đổi
  useEffect(() => {
    if (selectedRepoId) {
      fetchBranches();
    } else {
      setBranches([]);
      setSelectedBranch('');
    }
  }, [selectedRepoId]);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [branches, setBranches] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [branchLoading, setBranchLoading] = useState(false);
  const [daysBack, setDaysBack] = useState(30);

  // Fetch branches - CHỈ GỌI KHI USER CHỌN REPO
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

  // Fetch comprehensive analytics - CHỈ GỌI KHI USER CHỌN REPO VÀ NHẤN NÚT
  const fetchAnalytics = async () => {
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
  };

  const handleBranchChange = (branchName) => {
    setSelectedBranch(branchName);
    if (onBranchChange) {
      onBranchChange(branchName);
    }
  };

  const renderContent = () => {
    if (loading) return <div style={{textAlign: 'center', padding: '20px'}}><Spin /></div>;
    if (error) return <Alert message={error} type="error" showIcon />;
    if (!analyticsData) return <Typography.Text type="secondary">Chọn một repository để xem phân tích.</Typography.Text>;

    return (
      <div style={{ maxHeight: 'calc(100vh - 300px)', overflowY: 'auto', paddingRight: '8px' }}>
        <Collapse defaultActiveKey={['progress', 'risks']} ghost>
          <Collapse.Panel header={<><LineChartOutlined /> Tiến độ</>} key="progress">
            <ProgressSection data={analyticsData.progress} />
          </Collapse.Panel>
          <Collapse.Panel header={<><ExperimentOutlined /> Rủi ro</>} key="risks">
            <RisksSection data={analyticsData.risks} />
          </Collapse.Panel>
          <Collapse.Panel header={<><TeamOutlined /> Năng suất</>} key="productivity">
            <ProductivitySection data={analyticsData.productivity_metrics} />
          </Collapse.Panel>
          <Collapse.Panel header={<><UserSwitchOutlined /> Gợi ý phân công</>} key="assignment">
            <AssignmentSection data={analyticsData.assignment_suggestions} />
          </Collapse.Panel>
        </Collapse>
      </div>
    );
  };

  return (
    <SidebarCard
      title={<SectionTitle level={5} style={{ fontSize: '14px' }}>Bảng phân tích AI</SectionTitle>}
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
        </Space>
      )}
      
      {renderContent()}
    </SidebarCard>
  );
};

export default DashboardAnalyst;