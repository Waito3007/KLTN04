import React from 'react';
import { 
  Modal, Form, Input, Select, DatePicker, Button, Space, Avatar, Tag, 
  Card, Divider, Typography, Row, Col 
} from 'antd';
import { 
  UserOutlined, CalendarOutlined, FlagOutlined, 
  FileTextOutlined, TeamOutlined 
} from '@ant-design/icons';
import { getAvatarUrl } from '../../../utils/taskUtils.jsx';

const { Option } = Select;
const { TextArea } = Input;
const { Title, Text } = Typography;

const TaskModal = ({
  isModalVisible,
  editingTask,
  form,
  handleTaskSubmit,
  setIsModalVisible,
  collaborators
}) => {
  console.log('🎯 TaskModal rendered with collaborators:', collaborators);
  console.log('🎯 TaskModal collaborators type:', typeof collaborators);
  console.log('🎯 TaskModal collaborators isArray:', Array.isArray(collaborators));

  // Get current user and ensure they're in the assignee list
  const getCurrentUser = () => {
    try {
      const profile = JSON.parse(localStorage.getItem('github_profile') || '{}');
      return {
        login: profile.login,
        github_username: profile.login,
        avatar_url: profile.avatar_url,
        display_name: profile.name || profile.login,
        is_current_user: true
      };
    } catch {
      return null;
    }
  };

  const currentUser = getCurrentUser();
  
  // Combine current user with collaborators, avoiding duplicates
  const allAssignees = (() => {
    const assignees = Array.isArray(collaborators) ? [...collaborators] : [];
    
    if (currentUser && !assignees.some(c => c.login === currentUser.login)) {
      assignees.unshift(currentUser); // Add current user at the beginning
    }
    
    return assignees;
  })();
    return (
    <Modal
      title={null}
      open={isModalVisible}
      onCancel={() => setIsModalVisible(false)}
      footer={null}
      width={600}
      style={{ top: 20 }}
    >
      <div style={{ padding: '20px 0' }}>
        {/* Modal Header */}
        <div style={{ 
          textAlign: 'center', 
          marginBottom: 30,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          margin: '-24px -24px 30px -24px',
          padding: '20px 24px',
          borderRadius: '8px 8px 0 0'
        }}>
          <Title level={3} style={{ color: 'white', margin: 0 }}>
            {editingTask ? "✏️ Chỉnh sửa Task" : "➕ Tạo Task Mới"}
          </Title>
          <Text style={{ color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
            {editingTask ? "Cập nhật thông tin task" : "Tạo task mới cho dự án"}
          </Text>
        </div>

        <Form
          form={form}
          layout="vertical"
          onFinish={handleTaskSubmit}
          size="large"
        >
          {/* Task Title */}
          <Card 
            size="small" 
            title={
              <Space>
                <FileTextOutlined style={{ color: '#1890ff' }} />
                <span>Thông tin cơ bản</span>
              </Space>
            }
            style={{ marginBottom: 20, borderRadius: 8 }}
          >
            <Form.Item
              name="title"
              label="Tiêu đề Task"
              rules={[{ required: true, message: 'Vui lòng nhập tiêu đề!' }]}
            >
              <Input 
                placeholder="Nhập tiêu đề task..." 
                style={{ borderRadius: 6 }}
              />
            </Form.Item>
            
            <Form.Item
              name="description"
              label="Mô tả chi tiết"
              rules={[{ required: true, message: 'Vui lòng nhập mô tả!' }]}
            >
              <TextArea 
                rows={4} 
                placeholder="Mô tả chi tiết về task này..."
                style={{ borderRadius: 6 }}
              />
            </Form.Item>
          </Card>

          {/* Assignment & Priority */}
          <Card 
            size="small" 
            title={
              <Space>
                <TeamOutlined style={{ color: '#52c41a' }} />
                <span>Phân công & Ưu tiên</span>
              </Space>
            }
            style={{ marginBottom: 20, borderRadius: 8 }}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="assignee"
                  label="Giao cho"
                  rules={[{ required: true, message: 'Vui lòng chọn người thực hiện!' }]}
                >
                  <Select 
                    placeholder="Chọn thành viên..."
                    showSearch
                    optionFilterProp="children"
                    style={{ borderRadius: 6 }}
                    filterOption={(input, option) =>
                      option.children.props.children[1].toLowerCase().indexOf(input.toLowerCase()) >= 0
                    }                  >                    {allAssignees.map((collab, index) => {
                      const username = collab.login || collab.github_username;
                      const uniqueKey = username ? `${username}-${index}` : `unknown-${index}`;
                      
                      return (
                        <Option 
                          key={uniqueKey}
                          value={username}
                        >
                          <Space>
                            <Avatar src={getAvatarUrl(collab.avatar_url, username)} size="small" />
                            <span>{collab.display_name || username || 'Unknown User'}</span>
                            {collab.is_current_user && <Tag color="green" size="small">Bản thân</Tag>}
                            {collab.is_owner && <Tag color="gold" size="small">Owner</Tag>}
                            {collab.role && !collab.is_owner && <Tag color="blue" size="small">{collab.role}</Tag>}
                          </Space>
                        </Option>
                      );
                    })}
                  </Select>
                </Form.Item>
              </Col>
              
              <Col span={12}>
                <Form.Item
                  name="priority"
                  label="Độ ưu tiên"
                  rules={[{ required: true, message: 'Vui lòng chọn độ ưu tiên!' }]}
                >
                  <Select placeholder="Chọn độ ưu tiên..." style={{ borderRadius: 6 }}>
                    <Option value="low">
                      <Space>
                        <FlagOutlined style={{ color: '#52c41a' }} />
                        <Tag color="#52c41a">Thấp</Tag>
                      </Space>
                    </Option>
                    <Option value="medium">
                      <Space>
                        <FlagOutlined style={{ color: '#fa8c16' }} />
                        <Tag color="#fa8c16">Trung bình</Tag>
                      </Space>
                    </Option>
                    <Option value="high">
                      <Space>
                        <FlagOutlined style={{ color: '#f5222d' }} />
                        <Tag color="#f5222d">Cao</Tag>
                      </Space>
                    </Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>
          </Card>

          {/* Due Date */}
          <Card 
            size="small" 
            title={
              <Space>
                <CalendarOutlined style={{ color: '#fa8c16' }} />
                <span>Thời gian</span>
              </Space>
            }
            style={{ marginBottom: 20, borderRadius: 8 }}
          >
            <Form.Item
              name="dueDate"
              label="Hạn hoàn thành"
            >
              <DatePicker 
                style={{ width: '100%', borderRadius: 6 }} 
                placeholder="Chọn ngày hết hạn..."
              />
            </Form.Item>
          </Card>

          {/* Action Buttons */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'flex-end', 
            gap: 12,
            marginTop: 30,
            paddingTop: 20,
            borderTop: '1px solid #f0f0f0'
          }}>
            <Button 
              size="large"
              onClick={() => setIsModalVisible(false)}
              style={{ minWidth: 100, borderRadius: 6 }}
            >
              Hủy bỏ
            </Button>
            <Button 
              type="primary" 
              htmlType="submit"
              size="large"
              style={{ 
                minWidth: 120, 
                borderRadius: 6,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none'
              }}
            >
              {editingTask ? '💾 Cập nhật' : '🚀 Tạo Task'}
            </Button>
          </div>
        </Form>
      </div>
    </Modal>
  );
};

export default TaskModal;
