// frontend/src/components/Dashboard/AssignmentRecommendation.jsx
import React, { useState, useEffect } from 'react';
import { Card, Button, List, Avatar, Tag, Tooltip, Modal, Form, Select, Input, Spin, Alert } from 'antd';
import { UserOutlined, TrophyOutlined, ThunderboltOutlined, TeamOutlined } from '@ant-design/icons';
import { assignmentRecommendationAPI } from '../../services/api';

const { Option } = Select;

const AssignmentRecommendation = ({ repositoryId, onAssign }) => {
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [memberSkills, setMemberSkills] = useState([]);
  const [teamInsights, setTeamInsights] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

  const taskTypes = [
    { value: 'feat', label: '✨ Feature', color: 'blue' },
    { value: 'fix', label: '🐛 Bug Fix', color: 'red' },
    { value: 'docs', label: '📝 Documentation', color: 'green' },
    { value: 'refactor', label: '♻️ Refactor', color: 'orange' },
    { value: 'test', label: '✅ Test', color: 'purple' },
    { value: 'chore', label: '🔧 Chore', color: 'gray' },
    { value: 'style', label: '💄 Style', color: 'pink' },
    { value: 'perf', label: '⚡ Performance', color: 'yellow' }
  ];

  const taskAreas = [
    { value: 'frontend', label: '🎨 Frontend', color: 'blue' },
    { value: 'backend', label: '⚙️ Backend', color: 'green' },
    { value: 'database', label: '🗄️ Database', color: 'red' },
    { value: 'devops', label: '🚀 DevOps', color: 'orange' },
    { value: 'mobile', label: '📱 Mobile', color: 'purple' },
    { value: 'docs', label: '📚 Documentation', color: 'cyan' },
    { value: 'general', label: '🔧 General', color: 'gray' }
  ];

  const riskLevels = [
    { value: 'low', label: '🟢 Low Risk', color: 'green' },
    { value: 'medium', label: '🟡 Medium Risk', color: 'orange' },
    { value: 'high', label: '🔴 High Risk', color: 'red' }
  ];

  const priorities = [
    { value: 'LOW', label: 'Low', color: 'gray' },
    { value: 'MEDIUM', label: 'Medium', color: 'blue' },
    { value: 'HIGH', label: 'High', color: 'orange' },
    { value: 'URGENT', label: 'Urgent', color: 'red' }
  ];

  useEffect(() => {
    if (repositoryId) {
      loadMemberSkills();
      loadTeamInsights();
    }
  }, [repositoryId]);

  const loadMemberSkills = async () => {
    try {
      setLoading(true);
      const response = await assignmentRecommendationAPI.getMemberSkills(repositoryId);
      setMemberSkills(response.data);
    } catch (error) {
      console.error('Error loading member skills:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadTeamInsights = async () => {
    try {
      const response = await assignmentRecommendationAPI.getTeamInsights(repositoryId);
      setTeamInsights(response.data);
    } catch (error) {
      console.error('Error loading team insights:', error);
    }
  };

  const handleGetRecommendations = async (values) => {
    try {
      setLoading(true);
      const response = await assignmentRecommendationAPI.getRecommendations(repositoryId, {
        task_type: values.taskType,
        task_area: values.taskArea,
        risk_level: values.riskLevel,
        priority: values.priority,
        required_skills: values.requiredSkills ? values.requiredSkills.split(',').map(s => s.trim()) : null
      });
      setRecommendations(response.data);
      setModalVisible(false);
    } catch (error) {
      console.error('Error getting recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskToleranceColor = (tolerance) => {
    switch (tolerance) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'green';
      default: return 'gray';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return '#52c41a';
    if (score >= 60) return '#faad14';
    if (score >= 40) return '#fa8c16';
    return '#f5222d';
  };

  return (
    <div className="assignment-recommendation">
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3><TeamOutlined /> Đề xuất phân công thành viên</h3>
        <Button type="primary" onClick={() => setModalVisible(true)}>
          Tìm người phù hợp
        </Button>
      </div>

      {/* Team Insights */}
      {teamInsights && (
        <Card title={<><TrophyOutlined /> Team Insights</>} style={{ marginBottom: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
            <div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                {teamInsights.team_size}
              </div>
              <div style={{ color: '#666' }}>Active Members</div>
            </div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
                {teamInsights.total_commits_analyzed}
              </div>
              <div style={{ color: '#666' }}>Total Commits</div>
            </div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#faad14' }}>
                {teamInsights.area_coverage.covered_areas.length}
              </div>
              <div style={{ color: '#666' }}>Coverage Areas</div>
            </div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa541c' }}>
                {teamInsights.workload_summary.members_with_active_tasks}
              </div>
              <div style={{ color: '#666' }}>Members with Tasks</div>
            </div>
          </div>

          <div style={{ marginTop: 16 }}>
            <h4>Risk Tolerance Distribution:</h4>
            <div style={{ display: 'flex', gap: 8 }}>
              {Object.entries(teamInsights.risk_tolerance_distribution).map(([risk, count]) => (
                <Tag key={risk} color={getRiskToleranceColor(risk)}>
                  {risk.toUpperCase()}: {count}
                </Tag>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 16 }}>
            <h4>Area Experts:</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {Object.entries(teamInsights.area_coverage.area_experts).map(([area, experts]) => (
                <Tooltip key={area} title={`Experts: ${experts.join(', ')}`}>
                  <Tag color="blue">{area} ({experts.length})</Tag>
                </Tooltip>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Member Skills */}
      <Card title={<><UserOutlined /> Member Skills Analysis</>} style={{ marginBottom: 16 }}>
        <List
          loading={loading}
          dataSource={memberSkills}
          renderItem={member => (
            <List.Item>
              <List.Item.Meta
                avatar={<Avatar icon={<UserOutlined />} />}
                title={
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>{member.member}</span>
                    <Tag color={getRiskToleranceColor(member.risk_tolerance)}>
                      {member.risk_tolerance.toUpperCase()} RISK
                    </Tag>
                    <Tag color="blue">{member.total_commits} commits</Tag>
                  </div>
                }
                description={
                  <div>
                    <div style={{ marginBottom: 4 }}>
                      <strong>Expertise:</strong> {member.expertise_areas.join(', ') || 'General'}
                    </div>
                    <div style={{ marginBottom: 4 }}>
                      <strong>Top Types:</strong> {Object.entries(member.top_commit_types).slice(0, 3).map(([type, count]) => (
                        <Tag key={type} size="small">{type}: {count}</Tag>
                      ))}
                    </div>
                    <div>
                      <strong>Activity Score:</strong> 
                      <span style={{ color: member.recent_activity_score > 5 ? '#52c41a' : '#faad14', marginLeft: 4 }}>
                        {member.recent_activity_score}
                      </span>
                    </div>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card title={<><ThunderboltOutlined /> Đề xuất phân công</>}>
          <List
            dataSource={recommendations}
            renderItem={(rec, index) => (
              <List.Item
                actions={[
                  <Button 
                    type="primary" 
                    size="small"
                    onClick={() => onAssign && onAssign(rec.member)}
                  >
                    Assign
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <Avatar 
                      style={{ 
                        backgroundColor: getScoreColor(rec.adjusted_score || rec.score),
                        fontSize: '12px'
                      }}
                    >
                      #{index + 1}
                    </Avatar>
                  }
                  title={
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span>{rec.member}</span>
                      <Tag color="green">
                        Score: {Math.round(rec.adjusted_score || rec.score)}
                      </Tag>
                      {rec.workload_info && (
                        <Tag color="orange">
                          {rec.workload_info.active_tasks} active tasks
                        </Tag>
                      )}
                    </div>
                  }
                  description={
                    <div>
                      <div style={{ marginBottom: 8 }}>{rec.explanation}</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        Total commits: {rec.profile_summary.total_commits} | 
                        Risk tolerance: {rec.profile_summary.risk_tolerance} | 
                        Activity: {rec.profile_summary.recent_activity_score}
                      </div>
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* Recommendation Modal */}
      <Modal
        title="Tìm người phù hợp cho task"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleGetRecommendations}
        >
          <Form.Item
            name="taskType"
            label="Loại task"
            rules={[{ required: true, message: 'Vui lòng chọn loại task' }]}
          >
            <Select placeholder="Chọn loại task">
              {taskTypes.map(type => (
                <Option key={type.value} value={type.value}>
                  <Tag color={type.color}>{type.label}</Tag>
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="taskArea"
            label="Phạm vi công việc"
            rules={[{ required: true, message: 'Vui lòng chọn phạm vi' }]}
          >
            <Select placeholder="Chọn phạm vi công việc">
              {taskAreas.map(area => (
                <Option key={area.value} value={area.value}>
                  <Tag color={area.color}>{area.label}</Tag>
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="riskLevel"
            label="Mức độ rủi ro"
            rules={[{ required: true, message: 'Vui lòng chọn mức độ rủi ro' }]}
          >
            <Select placeholder="Chọn mức độ rủi ro">
              {riskLevels.map(risk => (
                <Option key={risk.value} value={risk.value}>
                  <Tag color={risk.color}>{risk.label}</Tag>
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="priority"
            label="Độ ưu tiên"
            initialValue="MEDIUM"
          >
            <Select>
              {priorities.map(priority => (
                <Option key={priority.value} value={priority.value}>
                  <Tag color={priority.color}>{priority.label}</Tag>
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="requiredSkills"
            label="Kỹ năng yêu cầu (optional)"
            help="Nhập các kỹ năng cách nhau bằng dấu phẩy. VD: Python, JavaScript, React"
          >
            <Input placeholder="Python, JavaScript, React..." />
          </Form.Item>

          <Form.Item>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <Button onClick={() => setModalVisible(false)}>
                Hủy
              </Button>
              <Button type="primary" htmlType="submit" loading={loading}>
                Tìm đề xuất
              </Button>
            </div>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default AssignmentRecommendation;
