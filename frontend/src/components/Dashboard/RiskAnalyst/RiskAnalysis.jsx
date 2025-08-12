import React from 'react';
import { Card, Typography, Empty, Tag, List, Select, Switch } from 'antd';
import { Pie } from 'react-chartjs-2';
import Widget from '@components/common/Widget';
import { Loading } from '@components/common';

const { Text } = Typography;

const RiskAnalysis = ({
  riskAnalysis,
  riskLoading,
  selectedMemberRisk,
  setSelectedMemberRisk,
  compareRiskMode,
  setCompareRiskMode,
  compareMemberRisk,
  setCompareMemberRisk,
  selectedBranch, // Thêm prop để hiển thị branch hiện tại
}) => {
  if (riskLoading) {
    return (
      <div style={{ textAlign: 'center', marginBottom: 16 }}>
        <Loading variant="circle" size="small" message="Đang tải phân tích rủi ro..." />
      </div>
    );
  }

  if (!riskAnalysis || !riskAnalysis.success || !riskAnalysis.risk_distribution) {
    return (
      <Empty
        description={riskAnalysis && !riskAnalysis.success ? `API Error: ${riskAnalysis.message || 'Unknown error'}` : 'Không có dữ liệu rủi ro.'}
      />
    );
  }

  const renderPieChart = (memberLogin) => {
    const isAll = !memberLogin;
    const summary = isAll
      ? riskAnalysis.risk_distribution
      : riskAnalysis.members_risk_analysis.find(m => m.member_login === memberLogin)?.risk_summary;

    if (!summary) return <Empty description="Không có dữ liệu." />;

    const labels = ['lowrisk', 'highrisk', 'unknown'];
    const data = labels.map(label => summary[label] || 0);

    return (
      <div style={{ width: 320 }}>
        <Text strong>{memberLogin || 'Tất cả'}</Text>
        <Pie
          data={{
            labels,
            datasets: [
              {
                data,
                backgroundColor: ['#22c55e', '#ef4444', '#64748b'],
              },
            ],
          }}
          options={{
            plugins: {
              legend: { position: 'right', labels: { font: { size: 14 } } },
            },
          }}
        />
        <List
          dataSource={labels.map(label => [label, summary[label] || 0])}
          renderItem={([label, value]) => (
            <List.Item>
              <Tag color={label === 'lowrisk' ? 'green' : label === 'highrisk' ? 'red' : 'default'} style={{ fontSize: 15 }}>
                {label}
              </Tag>
              : <Text>{value}</Text>
            </List.Item>
          )}
          style={{ paddingLeft: 0 }}
        />
        <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text type="secondary">Tổng số commit: {summary.total_commits || 0}</Text>
          {riskAnalysis?.branch_name && (
            <Tag color="blue" size="small">
              Branch: {riskAnalysis.branch_name}
            </Tag>
          )}
        </div>
      </div>
    );
  };

  return (
    <Card
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Text strong>Phân tích rủi ro</Text>
          {selectedBranch && (
            <Tag color="blue" style={{ marginLeft: 8 }}>
              📊 Branch: {selectedBranch}
            </Tag>
          )}
          {riskAnalysis?.branch_name && (
            <Tag color="green" style={{ fontSize: '12px' }}>
              🎯 Analyzed: {riskAnalysis.branch_name}
            </Tag>
          )}
        </div>
      }
      size="small"
      style={{ borderRadius: 16, boxShadow: '0 2px 8px rgba(239,68,68,0.08)' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
        <Text strong>Chọn thành viên:</Text>
        <Select
          style={{ minWidth: 180 }}
          value={selectedMemberRisk || ''}
          onChange={v => setSelectedMemberRisk(v)}
          allowClear
        >
          <Select.Option value="">Tất cả</Select.Option>
          {riskAnalysis.members_risk_analysis.map(m => (
            <Select.Option key={m.member_login} value={m.member_login}>
              {m.member_login}
            </Select.Option>
          ))}
        </Select>
        <Switch
          checked={compareRiskMode}
          onChange={setCompareRiskMode}
          checkedChildren="So sánh"
          unCheckedChildren="Xem đơn"
        />
        {compareRiskMode && (
          <>
            <Text strong>So với:</Text>
            <Select
              style={{ minWidth: 180 }}
              value={compareMemberRisk || ''}
              onChange={v => setCompareMemberRisk(v)}
              allowClear
            >
              <Select.Option value="">Tất cả</Select.Option>
              {riskAnalysis.members_risk_analysis.map(m => (
                <Select.Option key={m.member_login} value={m.member_login}>
                  {m.member_login}
                </Select.Option>
              ))}
            </Select>
          </>
        )}
      </div>
      
      {/* Branch Analysis Summary */}
      {riskAnalysis?.branch_name && (
        <div style={{ 
          marginBottom: 16, 
          padding: 12, 
          background: '#fef2f2', 
          borderRadius: 8,
          border: '1px solid #fecaca'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <Text strong style={{ color: '#dc2626' }}>🔍 Tóm tắt phân tích rủi ro cho branch:</Text>
            <Tag color="blue">{riskAnalysis.branch_name}</Tag>
          </div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <Text type="secondary">👥 Thành viên: {riskAnalysis.total_members}</Text>
            <Text type="secondary">📝 Commits phân tích: {riskAnalysis.total_commits_analyzed}</Text>
            <div style={{ display: 'flex', gap: 8 }}>
              <Tag color="green" size="small">Low: {riskAnalysis.risk_distribution?.lowrisk || 0}</Tag>
              <Tag color="red" size="small">High: {riskAnalysis.risk_distribution?.highrisk || 0}</Tag>
              <Tag color="default" size="small">Unknown: {riskAnalysis.risk_distribution?.unknown || 0}</Tag>
            </div>
          </div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 32 }}>
        {renderPieChart(selectedMemberRisk)}
        {compareRiskMode && renderPieChart(compareMemberRisk)}
      </div>
    </Card>
  );
};

export default RiskAnalysis;
