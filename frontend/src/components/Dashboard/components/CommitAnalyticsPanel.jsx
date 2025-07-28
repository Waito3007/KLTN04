import React from 'react';
import { Card, Tag, Typography, Row, Col, Empty, Spin } from 'antd';
import { BranchesOutlined, RobotOutlined } from '@ant-design/icons';
import ChartWrapper from './ChartWrapper';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const { Text, Title } = Typography;

/**
 * Component hiển thị thống kê và biểu đồ phân tích commit
 */
const CommitAnalyticsPanel = ({ memberCommits, selectedMember, selectedBranch, analysisLoading }) => {
  if (!selectedMember) {
    return (
      <Card>
        <Empty description="Chọn thành viên để xem phân tích commits" />
      </Card>
    );
  }

  // Chart data cho biểu đồ loại commit
  const chartData = memberCommits && memberCommits.statistics && memberCommits.statistics.commit_types ? {
    labels: Object.keys(memberCommits.statistics.commit_types),
    datasets: [{
      data: Object.values(memberCommits.statistics.commit_types),
      backgroundColor: [
        '#52c41a', '#f5222d', '#1890ff', '#fa8c16', 
        '#722ed1', '#13c2c2', '#eb2f96', '#666666',
        '#fadb14', '#a0d911', '#ff7a45', '#ff85c0'
      ]
    }]
  } : null;

  return (
    <Spin spinning={analysisLoading}>
      {memberCommits && memberCommits.summary && memberCommits.summary.total_commits === 0 ? (
        <Card>
          <Empty 
            description={
              <div>
                <p>Không tìm thấy commits cho @{selectedMember.login}</p>
                {selectedBranch && (
                  <p>trên nhánh <Tag color="blue">{selectedBranch}</Tag></p>
                )}
                {!selectedBranch && (
                  <p>trên tất cả các nhánh</p>
                )}
                <p>Thử:</p>
                <ul style={{ textAlign: 'left', margin: '0 auto', display: 'inline-block' }}>
                  <li>Chọn nhánh khác từ dropdown</li>
                  <li>Chọn "Tất cả nhánh" để xem toàn bộ</li>
                  <li>Kiểm tra tên người dùng có chính xác không</li>
                </ul>
              </div>
            }
          />
        </Card>
      ) : memberCommits && memberCommits.summary && memberCommits.statistics && memberCommits.commits ? (
        <>
          {/* Statistics Overview */}
          <Card 
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>📊 Phân tích commits - @{selectedMember.login}</span>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {selectedBranch && (
                    <Tag color="blue" icon={<BranchesOutlined />}>
                      Nhánh: {selectedBranch}
                    </Tag>
                  )}
                  {!selectedBranch && (
                    <Tag color="purple">
                      Tất cả nhánh
                    </Tag>
                  )}
                  {memberCommits.summary?.ai_powered && (
                    <Tag color="green" icon={<RobotOutlined />}>
                      🤖 {memberCommits.summary?.model_used || 'AI Powered'}
                    </Tag>
                  )}
                  {!memberCommits.summary?.ai_powered && (
                    <Tag color="orange">
                      📝 Dựa trên mẫu
                    </Tag>
                  )}
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {memberCommits.summary.analysis_date ? new Date(memberCommits.summary.analysis_date).toLocaleString() : ''}
                  </Text>
                </div>
              </div>
            } 
            style={{ marginBottom: '20px' }}
          >
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Title level={2} style={{ color: '#1890ff', margin: 0 }}>
                    {memberCommits.summary.total_commits}
                  </Title>
                  <Text>Tổng Commits</Text>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Title level={2} style={{ color: '#52c41a', margin: 0 }}>
                    +{memberCommits.statistics.productivity?.total_additions || 0}
                  </Title>
                  <Text>Dòng code thêm</Text>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Title level={2} style={{ color: '#f5222d', margin: 0 }}>
                    -{memberCommits.statistics.productivity?.total_deletions || 0}
                  </Title>
                  <Text>Dòng code xóa</Text>
                </div>
              </Col>
            </Row>
          </Card>

          <Row gutter={[16, 16]}>
            {/* Commit Types Chart */}
            <Col xs={24} lg={12}>
              <Card title="Loại Commit hiện tại" size="small">
                {chartData && (
                  <ChartWrapper
                    type="pie"
                    data={chartData}
                    options={{
                      plugins: {
                        legend: {
                          position: 'bottom',
                          labels: {
                            color: '#333',
                            font: {
                              size: 14,
                              weight: 'bold',
                            }
                          }
                        }
                      }
                    }}
                    style={{ height: '300px', display: 'flex', justifyContent: 'center' }}
                  />
                )}
              </Card>
            </Col>

            {/* Tech Areas Chart */}
            <Col xs={24} lg={12}>
              <Card title="🌐 Lĩnh vực công nghệ" size="small">
                {memberCommits.statistics.tech_analysis && (
                  <ChartWrapper
                    type="bar"
                    data={{
                      labels: Object.keys(memberCommits.statistics.tech_analysis),
                      datasets: [{
                        label: 'Số lượng',
                        data: Object.values(memberCommits.statistics.tech_analysis),
                        backgroundColor: '#1890ff',
                        borderColor: '#0056b3',
                        borderWidth: 2,
                      }]
                    }}
                    options={{
                      plugins: {
                        legend: {
                          position: 'top',
                          labels: {
                            color: '#333',
                            font: {
                              size: 14,
                              weight: 'bold',
                            }
                          }
                        }
                      }
                    }}
                    style={{ height: '300px', display: 'flex', justifyContent: 'center' }}
                  />
                )}
              </Card>
            </Col>
          </Row>

          
        </>
      ) : null}
    </Spin>
  );
};

export default CommitAnalyticsPanel;
