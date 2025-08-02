
import React from 'react';
import { Card, Typography, Spin, Empty, Tag, List, Select, Switch } from 'antd';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const { Text } = Typography;

const AreaAnalysis = ({
  areaAnalysis,
  areaLoading,
  selectedMemberArea,
  setSelectedMemberArea,
  compareAreaMode,
  setCompareAreaMode,
  compareMemberArea,
  setCompareMemberArea,
}) => {
  if (areaLoading) {
    return (
      <div style={{ textAlign: 'center', marginBottom: 16 }}>
        <Spin />
        <div style={{ marginTop: 8, color: '#666' }}>Đang tải phân tích lĩnh vực...</div>
      </div>
    );
  }

  if (!areaAnalysis || !areaAnalysis.success || !areaAnalysis.area_distribution) {
    return (
      <Empty
        description={areaAnalysis && !areaAnalysis.success ? `API Error: ${areaAnalysis.message || 'Unknown error'}` : 'Không có dữ liệu lĩnh vực.'}
      />
    );
  }

  const renderPieChart = (memberLogin) => {
    const isAll = !memberLogin;
    const summary = isAll
      ? areaAnalysis.area_distribution
      : areaAnalysis.members_area_analysis.find(m => m.member_login === memberLogin)?.area_summary;

    if (!summary) return <Empty description="Không có dữ liệu." />;

    const labels = Object.keys(summary).filter(k => k !== 'total_commits');
    const data = Object.values(summary).filter((v, i) => Object.keys(summary)[i] !== 'total_commits');

    return (
      <div style={{ width: 320 }}>
        <Text strong>{memberLogin || 'Tất cả'}</Text>
        <Pie
          data={{
            labels,
            datasets: [
              {
                data,
                backgroundColor: ['#8b5cf6', '#6366f1', '#3b82f6', '#06b6d4', '#f59e42', '#ef4444', '#22c55e'],
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
          dataSource={Object.entries(summary).filter(([k]) => k !== 'total_commits')}
          renderItem={([area, value]) => (
            <List.Item>
              <Tag color="purple" style={{ fontSize: 15 }}>
                {area}
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
      title={<Text strong>Phân tích lĩnh vực công nghệ</Text>}
      size="small"
      style={{ marginBottom: 32, borderRadius: 16, boxShadow: '0 2px 8px rgba(168,85,247,0.08)' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
        <Text strong>Chọn thành viên:</Text>
        <Select
          style={{ minWidth: 180 }}
          value={selectedMemberArea || ''}
          onChange={v => setSelectedMemberArea(v)}
          allowClear
        >
          <Select.Option value="">Tất cả</Select.Option>
          {areaAnalysis.members_area_analysis.map(m => (
            <Select.Option key={m.member_login} value={m.member_login}>
              {m.member_login}
            </Select.Option>
          ))}
        </Select>
        <Switch
          checked={compareAreaMode}
          onChange={setCompareAreaMode}
          checkedChildren="So sánh"
          unCheckedChildren="Xem đơn"
        />
        {compareAreaMode && (
          <>
            <Text strong>So với:</Text>
            <Select
              style={{ minWidth: 180 }}
              value={compareMemberArea || ''}
              onChange={v => setCompareMemberArea(v)}
              allowClear
            >
              <Select.Option value="">Tất cả</Select.Option>
              {areaAnalysis.members_area_analysis.map(m => (
                <Select.Option key={m.member_login} value={m.member_login}>
                  {m.member_login}
                </Select.Option>
              ))}
            </Select>
          </>
        )}
      </div>
      <div style={{ display: 'flex', gap: 32 }}>
        {renderPieChart(selectedMemberArea)}
        {compareAreaMode && renderPieChart(compareMemberArea)}
      </div>
    </Card>
  );
};

export default AreaAnalysis;
