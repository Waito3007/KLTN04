
import React from 'react';
import { Card, Typography, Spin, Empty, Tag, List, Select, Switch } from 'antd';
import { Pie } from 'react-chartjs-2';

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
}) => {
  if (riskLoading) {
    return (
      <div style={{ textAlign: 'center', marginBottom: 16 }}>
        <Spin />
        <div style={{ marginTop: 8, color: '#666' }}>Đang tải phân tích rủi ro...</div>
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
        <Text type="secondary">Tổng số commit: {summary.total_commits || 0}</Text>
      </div>
    );
  };

  return (
    <Card
      title={<Text strong>Phân tích rủi ro</Text>}
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
      <div style={{ display: 'flex', gap: 32 }}>
        {renderPieChart(selectedMemberRisk)}
        {compareRiskMode && renderPieChart(compareMemberRisk)}
      </div>
    </Card>
  );
};

export default RiskAnalysis;
