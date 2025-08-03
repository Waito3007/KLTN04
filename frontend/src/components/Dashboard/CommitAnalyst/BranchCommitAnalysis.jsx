
import React from 'react';
import { Card, Typography, Spin, Empty, Tag, List, Select } from 'antd';
import { Pie } from 'react-chartjs-2';

const { Text } = Typography;

const BranchCommitAnalysis = ({
  branchAnalysis,
  branchAnalysisLoading,
  branchAnalysisError,
  selectedMember,
  setSelectedMember,
  renderCommitList,
}) => {
  if (branchAnalysisLoading) {
    return <Spin />;
  }

  if (branchAnalysisError) {
    return <Empty description={branchAnalysisError} />;
  }

  if (!branchAnalysis || !branchAnalysis.commits) {
    return <Empty description="Chưa có dữ liệu commit cho nhánh này." />;
  }

  let typeCount = {};
  let filtered = selectedMember
    ? branchAnalysis.commits.filter(c => (c.author_name || 'Không rõ') === selectedMember)
    : branchAnalysis.commits;
  
  // Debug logging
  console.log('🔍 BranchCommitAnalysis Debug:', {
    totalCommits: branchAnalysis.commits?.length || 0,
    selectedMember,
    filteredCommits: filtered.length,
    summary: branchAnalysis.summary
  });
  
  filtered.forEach(c => {
    const type = c.analysis?.type || 'other';
    typeCount[type] = (typeCount[type] || 0) + 1;
  });
  
  console.log('📊 Commit Type Count:', typeCount);
  const pieLabels = Object.keys(typeCount);
  const pieValues = Object.values(typeCount);

  return (
    <Card
      title={<Text strong>Phân tích loại commit theo nhánh</Text>}
      size="small"
      style={{
        marginBottom: 32,
        borderRadius: 28,
        boxShadow: '0 4px 24px rgba(59,130,246,0.12)',
        background: '#f6f8fc',
        border: '1px solid #e0e7ef',
      }}
    >
      <div
        style={{
          width: '100%',
          background: '#fff',
          borderRadius: 24,
          boxShadow: '0 2px 12px rgba(59,130,246,0.08)',
          padding: 24,
          display: 'flex',
          flexDirection: 'row',
          gap: 32,
          alignItems: 'flex-start',
          justifyContent: 'center',
        }}
      >
        <div style={{ flex: 1, minWidth: 320 }}>
          <Text strong style={{ fontSize: 16, marginBottom: 16, display: 'block' }}>
            Lọc theo thành viên:
          </Text>
          <Select
            style={{ minWidth: 180, marginBottom: 16 }}
            placeholder="Chọn thành viên"
            value={selectedMember}
            onChange={setSelectedMember}
            allowClear
            disabled={!branchAnalysis || !branchAnalysis.commits}
          >
            <Select.Option value="">Tất cả</Select.Option>
            {Array.from(new Set(branchAnalysis.commits.map(c => c.author_name || 'Không rõ'))).map(author => (
              <Select.Option key={author} value={author}>
                {author}
              </Select.Option>
            ))}
          </Select>
          <Text strong style={{ fontSize: 16, marginBottom: 8, display: 'block' }}>
            Loại commit / Số lượng:
          </Text>
          <List
            dataSource={Object.entries(typeCount)}
            renderItem={([type, count]) => (
              <List.Item>
                <Tag color="blue" style={{ fontSize: 15 }}>
                  {type}
                </Tag>
                : <Text>{count}</Text>
              </List.Item>
            )}
            style={{ paddingLeft: 0 }}
          />
        </div>
        <div style={{ flex: 1, minWidth: 320, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <Text strong style={{ fontSize: 16, marginBottom: 16 }}>
            Biểu đồ loại commit:
          </Text>
          {pieLabels.length === 0 ? (
            <Empty description="Không có dữ liệu." />
          ) : (
            <div style={{ width: 320, height: 320 }}>
              <Pie
                data={{
                  labels: pieLabels,
                  datasets: [
                    {
                      data: pieValues,
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
            </div>
          )}
        </div>
      </div>
      <div style={{ marginTop: 32, padding: '0 24px' }}>
        <Text strong style={{ fontSize: 16 }}>
          Danh sách commit:
        </Text>
        {renderCommitList(filtered)}
      </div>
    </Card>
  );
};

export default BranchCommitAnalysis;
