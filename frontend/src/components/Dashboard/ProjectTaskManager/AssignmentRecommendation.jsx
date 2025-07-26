import React, { useState, useEffect } from 'react';
import { 
  Card, Button, Space, Modal, List, Avatar, Tag, Typography, 
  Spin, Alert, Tooltip, Progress, Collapse, Row, Col, Divider,
  Input, Select, Form, message
} from 'antd';
import { 
  RobotOutlined, TeamOutlined, BulbOutlined, BarChartOutlined,
  UserOutlined, TrophyOutlined, WarningOutlined, CheckCircleOutlined,
  StarOutlined, FireOutlined, ThunderboltOutlined
} from '@ant-design/icons';
import { assignmentRecommendationAPI } from '../../../services/api';
import { getAvatarUrl } from '../../../utils/taskUtils.jsx';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;
const { TextArea } = Input;
const { Option } = Select;

const AssignmentRecommendation = ({ 
  selectedRepo, 
  isVisible, 
  onClose, 
  onSelectAssignee,
  currentTaskData = null 
}) => {
  const [loading, setLoading] = useState(false);
  const [teamInsights, setTeamInsights] = useState(null);
  const [memberSkills, setMemberSkills] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [taskDescription, setTaskDescription] = useState(currentTaskData?.description || '');
  const [requiredSkills, setRequiredSkills] = useState([]);
  const [activeTab, setActiveTab] = useState('insights'); // 'insights', 'recommend'

  // Available skills for selection
  const availableSkills = [
    'JavaScript', 'Python', 'React', 'Node.js', 'Database', 'API', 'Frontend', 'Backend',
    'Testing', 'DevOps', 'UI/UX', 'Documentation', 'Bug Fix', 'Feature', 'Refactoring',
    'Security', 'Performance', 'Mobile', 'Web Development', 'Data Analysis'
  ];

  useEffect(() => {
    if (isVisible && selectedRepo) {
      loadTeamInsights();
      loadMemberSkills();
    }
  }, [isVisible, selectedRepo]);

  useEffect(() => {
    if (currentTaskData) {
      setTaskDescription(currentTaskData.description || currentTaskData.title || '');
    }
  }, [currentTaskData]);

  const loadTeamInsights = async () => {
    try {
      setLoading(true);
      const insights = await assignmentRecommendationAPI.getTeamInsights(
        selectedRepo.owner.login, 
        selectedRepo.name
      );
      setTeamInsights(insights);
    } catch (error) {
      console.error('Error loading team insights:', error);
      message.error('Không thể tải thông tin team');
    } finally {
      setLoading(false);
    }
  };

  const loadMemberSkills = async () => {
    try {
      const skills = await assignmentRecommendationAPI.getMemberSkills(
        selectedRepo.owner.login, 
        selectedRepo.name
      );
      setMemberSkills(skills.members || []);
    } catch (error) {
      console.error('Error loading member skills:', error);
      message.error('Không thể tải kỹ năng thành viên');
    }
  };

  const getRecommendations = async () => {
    if (!taskDescription.trim()) {
      message.warning('Vui lòng nhập mô tả task');
      return;
    }

    try {
      setLoading(true);
      const result = await assignmentRecommendationAPI.getSmartAssignment(
        selectedRepo.owner.login,
        selectedRepo.name,
        taskDescription,
        requiredSkills,
        true // consider workload
      );
      setRecommendations(result.recommendations || []);
      setActiveTab('recommend');
    } catch (error) {
      console.error('Error getting recommendations:', error);
      message.error('Không thể lấy gợi ý phân công');
    } finally {
      setLoading(false);
    }
  };

  const getSkillColor = (skill) => {
    const colors = {
      'JavaScript': 'gold',
      'Python': 'blue',
      'React': 'cyan',
      'Node.js': 'green',
      'Database': 'purple',
      'API': 'orange',
      'Frontend': 'magenta',
      'Backend': 'volcano',
      'Testing': 'lime',
      'DevOps': 'red'
    };
    return colors[skill] || 'default';
  };

  const getSkillIcon = (skill) => {
    const icons = {
      'JavaScript': '⚡',
      'Python': '🐍',
      'React': '⚛️',
      'Database': '🗄️',
      'API': '🔌',
      'Frontend': '🎨',
      'Backend': '⚙️',
      'Testing': '🧪',
      'Bug Fix': '🐛',
      'Feature': '✨'
    };
    return icons[skill] || '💡';
  };

  const renderTeamInsights = () => {
    if (loading) return <Spin size="large" />;
    if (!teamInsights) return <Alert message="Không có dữ liệu team insights" type="info" />;

    return (
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Team Overview */}
        <Card size="small" title="📊 Tổng quan Team (AI-Enhanced)">
          <Row gutter={16}>
            <Col span={5}>
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                  {teamInsights.total_members || 0}
                </Title>
                <Text type="secondary">Thành viên</Text>
              </div>
            </Col>
            <Col span={5}>
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#52c41a' }}>
                  {teamInsights.active_members || 0}
                </Title>
                <Text type="secondary">Hoạt động</Text>
              </div>
            </Col>
            <Col span={5}>
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#fa8c16' }}>
                  {teamInsights.avg_commits_per_member?.toFixed(1) || 0}
                </Title>
                <Text type="secondary">Commit TB</Text>
              </div>
            </Col>
            <Col span={5}>
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#722ed1' }}>
                  {teamInsights.total_skills || 0}
                </Title>
                <Text type="secondary">Kỹ năng</Text>
              </div>
            </Col>
            <Col span={4}>
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#eb2f96' }}>
                  🤖
                </Title>
                <Text type="secondary">AI-Powered</Text>
              </div>
            </Col>
          </Row>
          
          {/* AI Analysis Summary */}
          {teamInsights.ai_analysis_summary && (
            <div style={{ marginTop: 16, padding: 12, backgroundColor: '#f6ffed', borderRadius: 4 }}>
              <Text strong style={{ color: '#52c41a' }}>🤖 AI Analysis Summary:</Text>
              <div style={{ marginTop: 8 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <Text type="secondary">
                      AI Coverage: {(teamInsights.ai_analysis_summary.avg_ai_coverage * 100).toFixed(0)}%
                    </Text>
                  </Col>
                  <Col span={8}>
                    <Text type="secondary">
                      Models Used: {teamInsights.ai_analysis_summary.models_used?.join(', ') || 'MultiFusion V2, Area, Risk'}
                    </Text>
                  </Col>
                  <Col span={8}>
                    <Text type="secondary">
                      AI Analyzed: {teamInsights.ai_analysis_summary.total_ai_commits || 0} commits
                    </Text>
                  </Col>
                </Row>
              </div>
            </div>
          )}
        </Card>

        {/* Top Skills */}
        {teamInsights.top_skills && teamInsights.top_skills.length > 0 && (
          <Card size="small" title="🏆 Kỹ năng hàng đầu">
            <Space wrap>
              {teamInsights.top_skills.slice(0, 10).map((skill) => (
                <Tag 
                  key={skill.skill} 
                  color={getSkillColor(skill.skill)}
                  style={{ margin: '2px', padding: '4px 8px', fontSize: '12px' }}
                >
                  {getSkillIcon(skill.skill)} {skill.skill} ({skill.member_count})
                </Tag>
              ))}
            </Space>
          </Card>
        )}

        {/* Member Skills List */}
        <Card size="small" title="👥 Kỹ năng từng thành viên (AI-Enhanced)">
          <List
            dataSource={memberSkills}
            renderItem={(member) => (
              <List.Item>
                <List.Item.Meta
                  avatar={<Avatar src={getAvatarUrl(member.avatar_url, member.username)} />}
                  title={
                    <Space>
                      <Text strong>{member.display_name || member.username}</Text>
                      <Tag color="blue">{member.total_commits} commits</Tag>
                      <Tag color="green">{member.skills?.length || 0} skills</Tag>
                      {/* AI Coverage indicator */}
                      {member.ai_coverage && (
                        <Tag color={member.ai_coverage > 0.5 ? "gold" : "orange"}>
                          🤖 AI: {(member.ai_coverage * 100).toFixed(0)}%
                        </Tag>
                      )}
                    </Space>
                  }
                  description={
                    <div>
                      <Space wrap style={{ marginBottom: 8 }}>
                        {member.skills?.slice(0, 8).map(skill => (
                          <Tag 
                            key={skill.skill} 
                            color={getSkillColor(skill.skill)}
                            style={{ fontSize: '11px' }}
                          >
                            {getSkillIcon(skill.skill)} {skill.skill} ({(skill.confidence || 0).toFixed(2)})
                          </Tag>
                        ))}
                        {member.skills?.length > 8 && (
                          <Text type="secondary">+{member.skills.length - 8} more...</Text>
                        )}
                      </Space>
                      
                      {/* Show AI analysis results if available */}
                      {member.ai_predictions && (
                        <div style={{ marginTop: 8 }}>
                          <Text type="secondary" style={{ fontSize: '11px' }}>
                            🤖 AI Analysis: 
                          </Text>
                          <div style={{ marginTop: 4 }}>
                            {/* Top AI-predicted commit types */}
                            {member.ai_predictions.commit_types && Object.entries(member.ai_predictions.commit_types)
                              .sort(([,a], [,b]) => b - a)
                              .slice(0, 3).map(([type, count]) => (
                              <Tag 
                                key={`ai-type-${type}`}
                                color="purple"
                                style={{ fontSize: '10px', margin: '1px' }}
                              >
                                {type}: {count}
                              </Tag>
                            ))}
                            
                            {/* Top AI-predicted areas */}
                            {member.ai_predictions.areas && Object.entries(member.ai_predictions.areas)
                              .sort(([,a], [,b]) => b - a)
                              .slice(0, 2).map(([area, count]) => (
                              <Tag 
                                key={`ai-area-${area}`}
                                color="cyan"
                                style={{ fontSize: '10px', margin: '1px' }}
                              >
                                {area}: {count}
                              </Tag>
                            ))}
                            
                            {/* Risk tolerance from AI */}
                            {member.risk_tolerance && (
                              <Tag 
                                color={member.risk_tolerance === 'high' ? 'red' : member.risk_tolerance === 'medium' ? 'orange' : 'green'}
                                style={{ fontSize: '10px', margin: '1px' }}
                              >
                                Risk: {member.risk_tolerance}
                              </Tag>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      </Space>
    );
  };

  const renderRecommendationForm = () => (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card size="small" title="🤖 Tạo gợi ý phân công (AI-Enhanced)">
        <Alert
          message="AI-Powered Assignment Recommendation"
          description="Hệ thống sử dụng 3 AI models: MultiFusion V2 (commit type), Area Analyst (work scope), Risk Analyst (risk level) để phân tích và đề xuất phân công chính xác."
          type="info"
          icon={<RobotOutlined />}
          style={{ marginBottom: 16 }}
          showIcon
        />
        <Form layout="vertical">
          <Form.Item label="Mô tả task" required>
            <TextArea
              value={taskDescription}
              onChange={(e) => setTaskDescription(e.target.value)}
              placeholder="Nhập mô tả chi tiết về task cần phân công..."
              rows={4}
            />
          </Form.Item>
          
          <Form.Item label="Kỹ năng yêu cầu">
            <Select
              mode="multiple"
              placeholder="Chọn kỹ năng cần thiết..."
              value={requiredSkills}
              onChange={setRequiredSkills}
              style={{ width: '100%' }}
            >
              {availableSkills.map(skill => (
                <Option key={skill} value={skill}>
                  {getSkillIcon(skill)} {skill}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item>
            <Button 
              type="primary" 
              icon={<RobotOutlined />}
              onClick={getRecommendations}
              loading={loading}
              size="large"
              style={{ width: '100%' }}
            >
              🎯 Lấy gợi ý phân công
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {recommendations.length > 0 && (
        <Card size="small" title="💡 Gợi ý phân công">
          <List
            dataSource={recommendations}
            renderItem={(rec, index) => (
              <List.Item
                actions={[
                  <Button 
                    type="primary" 
                    size="small"
                    onClick={() => {
                      onSelectAssignee(rec.username);
                      message.success(`Đã chọn ${rec.display_name} cho task này`);
                      onClose();
                    }}
                  >
                    ✅ Chọn
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <div style={{ position: 'relative' }}>
                      <Avatar src={getAvatarUrl(rec.avatar_url, rec.username)} size="large" />
                      <div style={{
                        position: 'absolute',
                        top: -5,
                        right: -5,
                        background: '#52c41a',
                        borderRadius: '50%',
                        padding: '2px 6px',
                        fontSize: '12px',
                        color: 'white',
                        fontWeight: 'bold'
                      }}>
                        #{index + 1}
                      </div>
                    </div>
                  }
                  title={
                    <Space direction="vertical" size="small">
                      <Space>
                        <Text strong style={{ fontSize: '16px' }}>
                          {rec.member || rec.display_name || rec.username}
                        </Text>
                        <Tag color="gold">
                          <StarOutlined /> Score: {(rec.score || 0).toFixed(2)}
                        </Tag>
                      </Space>
                      <Space wrap>
                        <Tag color="blue">
                          <ThunderboltOutlined /> {rec.profile_summary?.total_commits || rec.total_commits || 0} commits
                        </Tag>
                        {rec.adjusted_score && (
                          <Tag color="purple">
                            Adjusted: {rec.adjusted_score.toFixed(2)}
                          </Tag>
                        )}
                        {rec.workload_info && (
                          <Tag color="orange">
                            Workload: {rec.workload_info.workload_score?.toFixed(1) || 0}
                          </Tag>
                        )}
                        {/* Show AI coverage if available */}
                        {rec.profile_summary?.ai_coverage && (
                          <Tag color={rec.profile_summary.ai_coverage > 0.5 ? "gold" : "orange"}>
                            🤖 AI: {(rec.profile_summary.ai_coverage * 100).toFixed(0)}%
                          </Tag>
                        )}
                        {/* Show risk tolerance */}
                        {rec.profile_summary?.risk_tolerance && (
                          <Tag color={
                            rec.profile_summary.risk_tolerance === 'high' ? 'red' : 
                            rec.profile_summary.risk_tolerance === 'medium' ? 'orange' : 'green'
                          }>
                            Risk: {rec.profile_summary.risk_tolerance}
                          </Tag>
                        )}
                      </Space>
                    </Space>
                  }
                  description={
                    <div>
                      <Paragraph ellipsis={{ rows: 2, expandable: true }}>
                        <Text type="secondary">{rec.explanation || rec.reason || 'No explanation available'}</Text>
                      </Paragraph>
                      <Space wrap style={{ marginTop: 8 }}>
                        {/* Show expertise areas from profile_summary */}
                        {rec.profile_summary?.expertise_areas?.slice(0, 4).map((area, idx) => (
                          <Tag 
                            key={`${area}-${idx}`}
                            color="blue"
                            style={{ fontSize: '11px' }}
                          >
                            {getSkillIcon(area)} {area}
                          </Tag>
                        ))}
                        
                        {/* Show top commit types from AI analysis */}
                        {rec.profile_summary?.top_commit_types && Object.entries(rec.profile_summary.top_commit_types)
                          .slice(0, 3).map(([type, count]) => (
                          <Tag 
                            key={type}
                            color="green"
                            style={{ fontSize: '11px' }}
                          >
                            {type}: {count}
                          </Tag>
                        ))}
                        
                        {/* Show AI predictions if available */}
                        {rec.profile_summary?.ai_predictions && (
                          <div style={{ marginTop: 4, width: '100%' }}>
                            <Text type="secondary" style={{ fontSize: '10px' }}>🤖 AI Analysis:</Text>
                            <div style={{ marginTop: 2 }}>
                              {/* AI Commit Types */}
                              {rec.profile_summary.ai_predictions.commit_types && 
                                Object.entries(rec.profile_summary.ai_predictions.commit_types)
                                  .sort(([,a], [,b]) => b - a)
                                  .slice(0, 2).map(([type, count]) => (
                                  <Tag 
                                    key={`ai-${type}`}
                                    color="purple"
                                    style={{ fontSize: '10px', margin: '1px' }}
                                  >
                                    {type}: {count}
                                  </Tag>
                                ))}
                              
                              {/* AI Areas */}
                              {rec.profile_summary.ai_predictions.areas && 
                                Object.entries(rec.profile_summary.ai_predictions.areas)
                                  .sort(([,a], [,b]) => b - a)
                                  .slice(0, 2).map(([area, count]) => (
                                  <Tag 
                                    key={`ai-area-${area}`}
                                    color="cyan"
                                    style={{ fontSize: '10px', margin: '1px' }}
                                  >
                                    {area}: {count}
                                  </Tag>
                                ))}
                              
                              {/* AI Risk Analysis */}
                              {rec.profile_summary.ai_predictions.risks && 
                                Object.entries(rec.profile_summary.ai_predictions.risks)
                                  .sort(([,a], [,b]) => b - a)
                                  .slice(0, 2).map(([risk, count]) => (
                                  <Tag 
                                    key={`ai-risk-${risk}`}
                                    color={risk === 'highrisk' || risk === 'high' ? 'red' : 'green'}
                                    style={{ fontSize: '10px', margin: '1px' }}
                                  >
                                    {risk}: {count}
                                  </Tag>
                                ))}
                            </div>
                          </div>
                        )}
                        
                        {/* Fallback: show relevant_skills if available */}
                        {rec.relevant_skills?.slice(0, 6).map(skill => (
                          <Tag 
                            key={skill.skill} 
                            color={getSkillColor(skill.skill)}
                            style={{ fontSize: '11px' }}
                          >
                            {getSkillIcon(skill.skill)} {skill.skill} ({(skill.confidence || 0).toFixed(2)})
                          </Tag>
                        ))}
                      </Space>
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}
    </Space>
  );

  return (
    <Modal
      title={
        <Space>
          <RobotOutlined style={{ color: '#1890ff' }} />
          <span>🤖 AI Assignment Recommendation</span>
        </Space>
      }
      open={isVisible}
      onCancel={onClose}
      footer={null}
      width={900}
      style={{ top: 20 }}
    >
      <div style={{ marginBottom: 16 }}>
        <Space.Compact style={{ width: '100%' }}>
          <Button 
            type={activeTab === 'insights' ? 'primary' : 'default'}
            onClick={() => setActiveTab('insights')}
            style={{ flex: 1 }}
          >
            📊 Team Insights
          </Button>
          <Button 
            type={activeTab === 'recommend' ? 'primary' : 'default'}
            onClick={() => setActiveTab('recommend')}
            style={{ flex: 1 }}
          >
            🎯 Gợi ý phân công
          </Button>
        </Space.Compact>
      </div>

      {activeTab === 'insights' ? renderTeamInsights() : renderRecommendationForm()}
    </Modal>
  );
};

export default AssignmentRecommendation;
