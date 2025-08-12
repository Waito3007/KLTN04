import React from 'react';
import { Card, Typography, Empty, Tag, List, Select, Switch } from 'antd';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import Widget from "@components/common/Widget";
import { Loading } from '@components/common';

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
  selectedBranch, // Thêm prop để hiển thị branch hiện tại
}) => {
  if (areaLoading) {
    return (
      <div style={{ textAlign: 'center', marginBottom: 16 }}>
        <Loading variant="circle" size="small" message="Đang tải phân tích lĩnh vực..." />
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
        <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text type="secondary">Tổng số commit: {summary.total_commits || 0}</Text>
          {areaAnalysis?.branch_name && (
            <Tag color="blue" size="small">
              Branch: {areaAnalysis.branch_name}
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
          <Text strong>Phân tích lĩnh vực công nghệ</Text>
          {selectedBranch && (
            <Tag color="blue" style={{ marginLeft: 8 }}>
              📊 Branch: {selectedBranch}
            </Tag>
          )}
          {areaAnalysis?.branch_name && (
            <Tag color="green" style={{ fontSize: '12px' }}>
              🎯 Analyzed: {areaAnalysis.branch_name}
            </Tag>
          )}
        </div>
      }
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
      
      {/* Branch Analysis Summary */}
      {areaAnalysis?.branch_name && (
        <div style={{ 
          marginBottom: 16, 
          padding: 12, 
          background: '#f8f9fa', 
          borderRadius: 8,
          border: '1px solid #e9ecef'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <Text strong style={{ color: '#6366f1' }}>📊 Tóm tắt phân tích cho branch:</Text>
            <Tag color="blue">{areaAnalysis.branch_name}</Tag>
          </div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <Text type="secondary">👥 Thành viên: {areaAnalysis.total_members}</Text>
            <Text type="secondary">📝 Commits phân tích: {areaAnalysis.total_commits_analyzed}</Text>
            <Text type="secondary">🎯 Lĩnh vực: {Object.keys(areaAnalysis.area_distribution || {}).length}</Text>
          </div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 32 }}>
        {renderPieChart(selectedMemberArea)}
        {compareAreaMode && renderPieChart(compareMemberArea)}
      </div>
    </Card>
  );
};

export default AreaAnalysis;
