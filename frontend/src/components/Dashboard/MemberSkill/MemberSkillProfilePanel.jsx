import React, { useState, useEffect } from 'react';
import { Card, Typography, Spin, Empty, Select, Alert, Row, Col, Tag, Avatar } from 'antd';
import { UserOutlined, TrophyOutlined, RocketOutlined, SafetyOutlined } from '@ant-design/icons';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const { Title, Text } = Typography;
const { Option } = Select;

const MemberSkillProfilePanel = ({ repositories = [], selectedRepoId, selectedBranch }) => {
  const [loading, setLoading] = useState(false);
  const [memberProfiles, setMemberProfiles] = useState([]);
  const [selectedMember, setSelectedMember] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch member skill profiles
    const fetchMemberProfiles = async () => {
      if (!selectedRepoId || !selectedBranch) return;

      setLoading(true);
      setError(null);

      try {
        const selectedRepo = repositories.find(r => r.id === selectedRepoId);
        if (!selectedRepo) {
          setError('Repository not found');
          return;
        }

        // Normalize owner name - xử lý trường hợp owner là object
        const ownerName = typeof selectedRepo.owner === 'string' 
          ? selectedRepo.owner 
          : selectedRepo.owner?.login || selectedRepo.owner?.name || 'unknown';

        console.log('🔍 MemberSkillProfilePanel: Fetching member profiles for:', {
          owner: ownerName,
          repo: selectedRepo.name,
          branch: selectedBranch
        });

        // TODO: Implement member skills API endpoint
        // For now, mock data to avoid 404 errors
        const mockData = {
          members: [
            {
              username: 'Waito3007',
              total_commits: 45,
              recent_activity_score: 85,
              risk_tolerance: 'medium',
              expertise_areas: ['Frontend', 'Backend', 'API Design'],
              commit_types: {
                feat: 20,
                fix: 15,
                docs: 5,
                test: 3,
                refactor: 2
              },
              areas: {
                frontend: 25,
                backend: 15,
                database: 3,
                testing: 2
              }
            },
            {
              username: 'Developer2',
              total_commits: 32,
              recent_activity_score: 72,
              risk_tolerance: 'high',
              expertise_areas: ['Testing', 'CI/CD'],
              commit_types: {
                feat: 12,
                fix: 8,
                test: 10,
                ci: 2
              },
              areas: {
                backend: 18,
                testing: 10,
                devops: 4
              }
            }
          ]
        };

        setMemberProfiles(mockData.members || []);
        
        // Auto-select first member
        if (mockData.members && mockData.members.length > 0) {
          setSelectedMember(mockData.members[0].username);
        }

        console.log('✅ MemberSkillProfilePanel: Mock member profiles loaded');

        /*
        // Real API call - implement when backend endpoint is ready
        const response = await fetch(
          `http://localhost:8000/api/assignment-recommendation/member-skills/${ownerName}/${selectedRepo.name}?branch_name=${encodeURIComponent(selectedBranch)}`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
              'Content-Type': 'application/json'
            }
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        setMemberProfiles(data.members || []);
        
        // Auto-select first member
        if (data.members && data.members.length > 0) {
          setSelectedMember(data.members[0].username);
        }
        */
      } catch (err) {
        console.error('Error fetching member profiles:', err);
        setError(err.message);
        setMemberProfiles([]);
      } finally {
        setLoading(false);
      }
    };

    if (selectedRepoId && selectedBranch) {
      fetchMemberProfiles();
    }
  }, [selectedRepoId, selectedBranch, repositories]);

  // Prepare radar chart data for selected member
  const getRadarChartData = () => {
    if (!selectedMember || !memberProfiles.length) return null;

    const member = memberProfiles.find(m => m.username === selectedMember);
    if (!member) return null;

    // Extract skill levels from commit types and areas
    const commitTypes = member.commit_types || {};
    const areas = member.areas || {};
    const totalCommits = member.total_commits || 1;

    // Normalize scores to 0-100 scale
    const skills = {
      'Feature Development': ((commitTypes.feat || 0) / totalCommits) * 100,
      'Bug Fixing': ((commitTypes.fix || 0) / totalCommits) * 100,
      'Documentation': ((commitTypes.docs || 0) / totalCommits) * 100,
      'Frontend': ((areas.frontend || 0) / totalCommits) * 100,
      'Backend': ((areas.backend || 0) / totalCommits) * 100,
      'Testing': ((commitTypes.test || 0) / totalCommits) * 100,
    };

    return {
      labels: Object.keys(skills),
      datasets: [
        {
          label: 'Skill Level (%)',
          data: Object.values(skills),
          backgroundColor: 'rgba(59, 130, 246, 0.2)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(59, 130, 246, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(59, 130, 246, 1)',
        },
      ],
    };
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        beginAtZero: true,
        max: 100,
        min: 0,
        ticks: {
          stepSize: 20,
          callback: function(value) {
            return value + '%';
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        angleLines: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        pointLabels: {
          font: {
            size: 12,
          },
        },
      },
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${Math.round(context.parsed.r)}%`;
          }
        }
      }
    },
  };

  const selectedMemberData = memberProfiles.find(m => m.username === selectedMember);

  if (loading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>Loading member skill profiles...</div>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Alert
          message="Error Loading Member Profiles"
          description={error}
          type="error"
          showIcon
        />
      </Card>
    );
  }

  if (!memberProfiles.length) {
    return (
      <Card>
        <Empty 
          description="No member profiles found"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      </Card>
    );
  }

  const radarData = getRadarChartData();

  return (
    <div>
      {/* Member Selection */}
      <div style={{ marginBottom: 16 }}>
        <Text strong>Select Member: </Text>
        <Select
          style={{ width: 200, marginLeft: 8 }}
          value={selectedMember}
          onChange={setSelectedMember}
          placeholder="Choose a member"
        >
          {memberProfiles.map(member => (
            <Option key={member.username} value={member.username}>
              <Avatar 
                size="small" 
                icon={<UserOutlined />} 
                style={{ marginRight: 8 }}
              />
              {member.username}
            </Option>
          ))}
        </Select>
      </div>

      {selectedMemberData && (
        <Row gutter={[16, 16]}>
          {/* Member Overview */}
          <Col xs={24} lg={8}>
            <Card title="Member Overview" size="small">
              <div style={{ textAlign: 'center', marginBottom: 16 }}>
                <Avatar 
                  size={64} 
                  icon={<UserOutlined />}
                  style={{ backgroundColor: '#1890ff' }}
                />
                <Title level={4} style={{ margin: '8px 0 4px' }}>
                  {selectedMemberData.username}
                </Title>
              </div>
              
              <div style={{ marginBottom: 8 }}>
                <TrophyOutlined style={{ color: '#faad14', marginRight: 4 }} />
                <Text strong>Total Commits: </Text>
                <Text>{selectedMemberData.total_commits}</Text>
              </div>
              
              <div style={{ marginBottom: 8 }}>
                <RocketOutlined style={{ color: '#52c41a', marginRight: 4 }} />
                <Text strong>Activity Score: </Text>
                <Text>{Math.round(selectedMemberData.recent_activity_score || 0)}</Text>
              </div>
              
              <div style={{ marginBottom: 8 }}>
                <SafetyOutlined style={{ color: '#722ed1', marginRight: 4 }} />
                <Text strong>Risk Tolerance: </Text>
                <Tag color={
                  selectedMemberData.risk_tolerance === 'high' ? 'red' :
                  selectedMemberData.risk_tolerance === 'medium' ? 'orange' : 'green'
                }>
                  {selectedMemberData.risk_tolerance}
                </Tag>
              </div>

              {/* Expertise Areas */}
              <div style={{ marginTop: 16 }}>
                <Text strong>Expertise Areas:</Text>
                <div style={{ marginTop: 8 }}>
                  {selectedMemberData.expertise_areas?.map(area => (
                    <Tag key={area} color="blue" style={{ marginBottom: 4 }}>
                      {area}
                    </Tag>
                  )) || <Text type="secondary">No specific expertise</Text>}
                </div>
              </div>
            </Card>
          </Col>

          {/* Skill Radar Chart */}
          <Col xs={24} lg={16}>
            <Card title="Skill Profile Radar" size="small">
              {radarData ? (
                <div style={{ height: '300px' }}>
                  <Radar data={radarData} options={radarOptions} />
                </div>
              ) : (
                <Empty description="No skill data available" />
              )}
            </Card>
          </Col>

          {/* Detailed Skills */}
          <Col xs={24}>
            <Card title="Detailed Skill Breakdown" size="small">
              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <Text strong>Commit Types:</Text>
                  <div style={{ marginTop: 8 }}>
                    {Object.entries(selectedMemberData.commit_types || {}).map(([type, count]) => (
                      <div key={type} style={{ marginBottom: 4 }}>
                        <Tag color="geekblue">{type}</Tag>
                        <Text>{count} commits</Text>
                      </div>
                    ))}
                  </div>
                </Col>
                <Col xs={24} md={12}>
                  <Text strong>Development Areas:</Text>
                  <div style={{ marginTop: 8 }}>
                    {Object.entries(selectedMemberData.areas || {}).map(([area, count]) => (
                      <div key={area} style={{ marginBottom: 4 }}>
                        <Tag color="purple">{area}</Tag>
                        <Text>{count} commits</Text>
                      </div>
                    ))}
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default MemberSkillProfilePanel;