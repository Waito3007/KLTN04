import React, { useState, useEffect, useCallback } from 'react';
import { 
  Card, Avatar, List, Tag, Progress, Row, Col, Button, 
  Typography, Divider, Spin, Empty, message, Space, Switch, Select 
} from 'antd';
import { 
  UserOutlined, RobotOutlined, CodeOutlined, 
  BugOutlined, ToolOutlined, FileTextOutlined, BranchesOutlined 
} from '@ant-design/icons';
import { Pie, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const { Title, Text } = Typography;

const RepositoryMembers = ({ selectedRepo }) => {  const [members, setMembers] = useState([]);
  const [selectedMember, setSelectedMember] = useState(null);
  const [memberCommits, setMemberCommits] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [showAIFeatures, setShowAIFeatures] = useState(false);
  const [useAI, setUseAI] = useState(true); // Toggle for AI analysis
  const [aiModelStatus, setAiModelStatus] = useState(null);
  const [branches, setBranches] = useState([]);
  const [selectedBranch, setSelectedBranch] = useState(null); // NEW: Branch selector
  const [branchesLoading, setBranchesLoading] = useState(false);
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
      const response = await fetch(`http://localhost:8000/api/repositories/${selectedRepo.id}/ai/model-status`);
      
      if (response.ok) {
        const data = await response.json();
        console.log('AI Model Status:', data); // Debug log
        setAiModelStatus(data);
      } else {
        console.error('AI Status Error:', response.status);
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
        console.log('Branches API Response:', data);
        setBranches(data.branches || []);        // Auto-select: Start with "All Branches" by default
        if (!selectedBranch) {
          setSelectedBranch(null); // null = "All Branches"
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
    } else {
      console.log('❌ No selectedRepo or selectedRepo.id found:', {
        selectedRepo: !!selectedRepo,
        id: selectedRepo?.id
      });
      // Clear members if no repo
      setMembers([]);
    }  }, [selectedRepo, loadRepositoryMembers, loadRepositoryBranches, _loadAIModelStatus]); // Remove selectedRepo?.id to avoid redundancy

  // Re-analyze when branch changes
  useEffect(() => {
    if (selectedMember && selectedBranch !== null) {
      console.log('Branch changed, re-analyzing member:', selectedMember.login, 'on branch:', selectedBranch);
      handleMemberClick(selectedMember);
    }
  }, [selectedBranch]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleMemberClick = async (member) => {
    setSelectedMember(member);
    setAnalysisLoading(true);
    
    try {
      const aiParam = useAI ? '?use_ai=true' : '?use_ai=false';
      const branchParam = selectedBranch ? `&branch_name=${encodeURIComponent(selectedBranch)}` : '';
      const response = await fetch(
        `http://localhost:8000/api/repositories/${selectedRepo.id}/members/${member.login}/commits${aiParam}${branchParam}`
      );
      
      if (response.ok) {
        const data = await response.json();
        console.log('Commit Analysis Response:', data); // Debug log
        setMemberCommits(data.data);
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

  const getCommitTypeIcon = (type) => {
    const icons = {
      'feat': <CodeOutlined style={{ color: '#52c41a' }} />,
      'fix': <BugOutlined style={{ color: '#f5222d' }} />,
      'chore': <ToolOutlined style={{ color: '#1890ff' }} />,
      'docs': <FileTextOutlined style={{ color: '#fa8c16' }} />,
      'refactor': '♻️',
      'test': '✅',
      'style': '💄',
      'other': '📝'
    };
    return icons[type] || '📝';
  };

  const getCommitTypeColor = (type) => {
    const colors = {
      'feat': 'green',
      'fix': 'red',
      'chore': 'blue',
      'docs': 'orange',
      'refactor': 'purple',
      'test': 'cyan',
      'style': 'magenta',
      'other': 'default'
    };
    return colors[type] || 'default';
  };

  // Chart data for commit types
  const chartData = memberCommits ? {
    labels: Object.keys(memberCommits.statistics.commit_types),
    datasets: [{
      data: Object.values(memberCommits.statistics.commit_types),
      backgroundColor: [
        '#52c41a', '#f5222d', '#1890ff', '#fa8c16', 
        '#722ed1', '#13c2c2', '#eb2f96', '#666666'
      ]
    }]
  } : null;

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
        <Space wrap>{/* Branch Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BranchesOutlined />
            <Text strong style={{ fontSize: '14px' }}>Nhánh:</Text>
            <Select
              value={selectedBranch}
              onChange={setSelectedBranch}
              placeholder="Chọn nhánh"
              style={{ minWidth: 150 }}
              loading={branchesLoading}
              allowClear
            >
              <Select.Option key="all" value={null}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Tag color="purple" size="small">Tất cả</Tag>
                  Tất cả nhánh
                </span>
              </Select.Option>
              {branches.map(branch => (
                <Select.Option key={branch.name} value={branch.name}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {branch.is_default && <Tag color="blue" size="small">Mặc định</Tag>}
                    {branch.name}
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      ({branch.commits_count} commits)
                    </Text>
                  </span>
                </Select.Option>
              ))}
            </Select>
          </div>
          
          {/* AI Toggle Switch */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Text>Phân tích mẫu</Text>
            <Switch 
              checked={useAI}
              onChange={setUseAI}
              checkedChildren="🤖 AI"
              unCheckedChildren="📝 Cơ bản"
              style={{
                backgroundColor: useAI ? '#52c41a' : '#d9d9d9'
              }}
            />
            <Text>Mô hình HAN AI</Text>
          </div>
          
          <Button 
            type="primary" 
            icon={<RobotOutlined />}
            onClick={() => setShowAIFeatures(!showAIFeatures)}
            style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              border: 'none'
            }}
          >
            🤖 AI Features
          </Button>        </Space>
      </div>

      {/* Thống kê tổng quan */}
      {members.length > 0 && (
        <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
          <Col xs={24} sm={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Title level={3} style={{ color: '#1890ff', margin: 0 }}>
                {members.length}
              </Title>
              <Text type="secondary">Thành viên tham gia</Text>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Title level={3} style={{ color: '#52c41a', margin: 0 }}>
                {branches.length}
              </Title>
              <Text type="secondary">Nhánh trong dự án</Text>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <Title level={3} style={{ color: '#fa8c16', margin: 0 }}>
                {members.reduce((sum, member) => sum + member.total_commits, 0)}
              </Title>
              <Text type="secondary">Tổng commits</Text>
            </Card>
          </Col>
        </Row>
      )}{/* AI Features Panel */}
      {showAIFeatures && (
        <Card 
          style={{ marginBottom: '20px', borderColor: '#1890ff' }}
          title={
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>🤖 AI Model Status</span>
              {aiModelStatus && (
                <Tag color={aiModelStatus.model_loaded ? 'green' : 'red'}>
                  {aiModelStatus.model_loaded ? '✅ Model Loaded' : '❌ Model Not Available'}
                </Tag>
              )}
            </div>
          }
        >
          {aiModelStatus ? (
            <>              <Text strong>Loại mô hình: </Text>
              <Text>{aiModelStatus.model_info?.type || 'Mạng HAN'}</Text>
              <br />
              <Text strong>Chế độ phân tích: </Text>
              <Text>{useAI ? 'Hỗ trợ AI (Mô hình HAN)' : 'Dựa trên mẫu (Dự phòng)'}</Text>
              <Divider />
            </>
          ) : null}
          
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <div style={{ textAlign: 'center' }}>
                <CodeOutlined style={{ fontSize: '24px', color: useAI ? '#52c41a' : '#d9d9d9' }} />                <div>Phân loại Commit</div>
                <Text type="secondary">feat/fix/chore/docs</Text>
              </div>
            </Col>
            <Col span={6}>
              <div style={{ textAlign: 'center' }}>
                <UserOutlined style={{ fontSize: '24px', color: useAI ? '#1890ff' : '#d9d9d9' }} />
                <div>Thông tin Developer</div>
                <Text type="secondary">Phân tích năng suất</Text>
              </div>
            </Col>
            <Col span={6}>
              <div style={{ textAlign: 'center' }}>
                <ToolOutlined style={{ fontSize: '24px', color: useAI ? '#fa8c16' : '#d9d9d9' }} />
                <div>Phát hiện lĩnh vực công nghệ</div>
                <Text type="secondary">API/Frontend/Database</Text>
              </div>
            </Col>
            <Col span={6}>
              <div style={{ textAlign: 'center' }}>
                <FileTextOutlined style={{ fontSize: '24px', color: useAI ? '#722ed1' : '#d9d9d9' }} />
                <div>Nhận dạng mẫu</div>
                <Text type="secondary">Mẫu thay đổi code</Text>
              </div>
            </Col>
          </Row>
        </Card>
      )}

      <Row gutter={[24, 24]}>
        {/* Members List */}
        <Col xs={24} md={8}>
          <Card title="👥 Danh sách thành viên" loading={loading}>
            {members.length === 0 ? (
              <Empty description="Không có thành viên nào" />
            ) : (
              <List
                dataSource={members}
                renderItem={member => (
                  <List.Item
                    style={{
                      cursor: 'pointer',
                      padding: '12px',
                      backgroundColor: selectedMember?.login === member.login ? '#e6f7ff' : 'transparent',
                      borderRadius: '6px',
                      marginBottom: '8px'
                    }}
                    onClick={() => handleMemberClick(member)}
                  >
                    <List.Item.Meta
                      avatar={
                        <Avatar 
                          src={member.avatar_url} 
                          icon={<UserOutlined />}
                          size="large"
                        />
                      }
                      title={member.display_name}
                      description={
                        <div>
                          <div>@{member.login}</div>
                          <Text type="secondary">{member.total_commits} commits</Text>
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </Card>
        </Col>

        {/* Member Analysis */}
        <Col xs={24} md={16}>
          {!selectedMember ? (
            <Card>
              <Empty description="Chọn thành viên để xem phân tích commits" />
            </Card>          ) : (
            <Spin spinning={analysisLoading}>
              {memberCommits && memberCommits.summary.total_commits === 0 ? (
                <Card>
                  <Empty 
                    description={
                      <div>                        <p>Không tìm thấy commits cho @{selectedMember.login}</p>
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
              ) : memberCommits && (
                <>{/* Statistics Overview */}
                  <Card 
                    title={                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>📊 Phân tích commits - @{selectedMember.login}</span>
                        <div style={{ display: 'flex', gap: '8px' }}>                          {selectedBranch && (
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
                              🤖 Hỗ trợ AI
                            </Tag>
                          )}
                          {!memberCommits.summary?.ai_powered && (
                            <Tag color="orange">
                              📝 Dựa trên mẫu
                            </Tag>
                          )}
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {new Date(memberCommits.summary.analysis_date).toLocaleString()}
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
                          </Title>                          <Text>Tổng Commits</Text>
                        </div>
                      </Col>
                      <Col span={8}>
                        <div style={{ textAlign: 'center' }}>
                          <Title level={2} style={{ color: '#52c41a', margin: 0 }}>
                            +{memberCommits.statistics.productivity.total_additions}
                          </Title>
                          <Text>Dòng code thêm</Text>
                        </div>
                      </Col>
                      <Col span={8}>
                        <div style={{ textAlign: 'center' }}>
                          <Title level={2} style={{ color: '#f5222d', margin: 0 }}>
                            -{memberCommits.statistics.productivity.total_deletions}
                          </Title>
                          <Text>Dòng code xóa</Text>
                        </div>
                      </Col>
                    </Row>
                  </Card>

                  <Row gutter={[16, 16]}>
                    {/* Commit Types Chart */}
                    <Col xs={24} lg={12}>                      <Card title="🏷️ Loại Commit" size="small">
                        {chartData && (
                          <div style={{ height: '300px', display: 'flex', justifyContent: 'center' }}>
                            <Pie 
                              data={chartData} 
                              options={{ 
                                responsive: true, 
                                maintainAspectRatio: false,
                                plugins: {
                                  legend: {
                                    position: 'bottom'
                                  }
                                }
                              }} 
                            />
                          </div>
                        )}
                      </Card>
                    </Col>

                    {/* Tech Areas */}
                    <Col xs={24} lg={12}>
                      <Card title="🛠️ Lĩnh vực công nghệ" size="small">
                        <div style={{ height: '300px' }}>
                          {Object.entries(memberCommits.statistics.tech_analysis).map(([tech, count]) => (
                            <div key={tech} style={{ marginBottom: '12px' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>{tech}</span>
                                <span>{count}</span>
                              </div>
                              <Progress 
                                percent={(count / memberCommits.summary.total_commits) * 100} 
                                size="small"
                                showInfo={false}
                              />
                            </div>
                          ))}
                        </div>
                      </Card>
                    </Col>
                  </Row>                  {/* Recent Commits */}
                  <Card title="📝 Commits gần đây" style={{ marginTop: '20px' }}>
                    <List
                      dataSource={memberCommits.commits.slice(0, 10)}
                      renderItem={commit => (
                        <List.Item>
                          <List.Item.Meta
                            title={
                              <div>
                                <span style={{ marginRight: '8px' }}>
                                  {commit.message.length > 80 ? 
                                    commit.message.substring(0, 80) + '...' : 
                                    commit.message
                                  }
                                </span>
                                <Tag 
                                  color={getCommitTypeColor(commit.analysis.type)}
                                  icon={getCommitTypeIcon(commit.analysis.type)}
                                >
                                  {commit.analysis.type_icon} {commit.analysis.type}
                                </Tag>
                                <Tag color="blue">{commit.analysis.tech_area}</Tag>
                                {commit.analysis.ai_powered && (
                                  <>                                    {commit.analysis.impact && (
                                      <Tag color={commit.analysis.impact === 'high' ? 'red' : 
                                                 commit.analysis.impact === 'medium' ? 'orange' : 'green'}>
                                        Tác động: {commit.analysis.impact === 'high' ? 'Cao' : 
                                                  commit.analysis.impact === 'medium' ? 'Trung bình' : 'Thấp'}
                                      </Tag>
                                    )}
                                    {commit.analysis.urgency && (
                                      <Tag color={commit.analysis.urgency === 'urgent' ? 'red' : 
                                                 commit.analysis.urgency === 'high' ? 'orange' : 'default'}>
                                        {commit.analysis.urgency === 'urgent' ? 'Khẩn cấp' : 
                                         commit.analysis.urgency === 'high' ? 'Cao' : commit.analysis.urgency}
                                      </Tag>
                                    )}
                                    <Tag color="green" style={{ fontSize: '10px' }}>
                                      🤖 AI
                                    </Tag>
                                  </>
                                )}
                              </div>
                            }
                            description={
                              <div>
                                <Text code>{commit.sha}</Text> •                                <Text type="secondary">
                                  {commit.date ? 
                                    new Date(commit.date).toLocaleDateString('vi-VN') : 
                                    'Ngày không xác định'
                                  }
                                </Text> • 
                                <Text style={{ color: '#52c41a' }}>+{commit.stats?.insertions || 0}</Text> 
                                <Text style={{ color: '#f5222d' }}> -{commit.stats?.deletions || 0}</Text>
                                {commit.stats?.files_changed && (
                                  <Text type="secondary"> • {commit.stats.files_changed} files</Text>
                                )}
                              </div>
                            }
                          />
                        </List.Item>
                      )}
                    />                  </Card>
                </>
              )}
            </Spin>
          )}
        </Col></Row>
    </div>
  );
};

export default RepositoryMembers;
