import React, { useState } from 'react';
import { 
  Modal, Form, Input, Select, DatePicker, Button, Space, Avatar, Tag, 
  Card, Divider, Typography, Row, Col 
} from 'antd';
import { 
  UserOutlined, CalendarOutlined, FlagOutlined, 
  FileTextOutlined, TeamOutlined, RobotOutlined 
} from '@ant-design/icons';
import { getAvatarUrl } from '../../../utils/taskUtils.jsx';
import AssignmentRecommendation from './AssignmentRecommendation';

import styles from './TaskModal.module.css';

const { Option } = Select;
const { TextArea } = Input;
const { Title, Text } = Typography;

const TaskModal = ({
  isModalVisible,
  editingTask,
  form,
  handleTaskSubmit,
  setIsModalVisible,
  collaborators,
  selectedRepo // Add selectedRepo prop for AI recommendations
}) => {
  const [showAIRecommendation, setShowAIRecommendation] = useState(false);
  
  console.log('🎯 TaskModal rendered with collaborators:', collaborators);
  console.log('🎯 TaskModal collaborators type:', typeof collaborators);
  console.log('🎯 TaskModal collaborators isArray:', Array.isArray(collaborators));

  // Handle AI recommendation selection
  const handleAIAssigneeSelect = (username) => {
    form.setFieldsValue({ assignee: username });
    setShowAIRecommendation(false);
  };

  // Get current task data for AI recommendations
  const getCurrentTaskData = () => {
    const formValues = form.getFieldsValue();
    return {
      title: formValues.title || '',
      description: formValues.description || '',
      ...editingTask
    };
  };

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
      title={
        <div className={styles.modalHeaderContent}>
          <Title level={3} className={styles.modalTitle}>
            {editingTask ? "✏️ Chỉnh sửa Task" : "➕ Tạo Task Mới"}
          </Title>
          <Text className={styles.modalSubtitle}>
            {editingTask ? "Cập nhật thông tin task" : "Tạo task mới cho dự án"}
          </Text>
        </div>
      }
      open={isModalVisible}
      onCancel={() => setIsModalVisible(false)}
      footer={null}
      width={600}
      className={styles.taskModal}
      styles={{
        header: { padding: 0, borderBottom: 'none' },
        body: { padding: '24px 24px 0 24px' },
        content: { borderRadius: 12, overflow: 'hidden' }
      }}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleTaskSubmit}
        size="large"
        className={styles.taskForm}
      >
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <FileTextOutlined className={styles.sectionIcon} />
            <Text strong>Thông tin cơ bản</Text>
          </div>
          <Form.Item
            name="title"
            label="Tiêu đề Task"
            rules={[{ required: true, message: 'Vui lòng nhập tiêu đề!' }]}
          >
            <Input
              placeholder="Nhập tiêu đề task..."
              className={styles.inputField}
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
              className={styles.inputField}
            />
          </Form.Item>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <TeamOutlined className={styles.sectionIcon} />
            <Text strong>Phân công & Ưu tiên</Text>
          </div>
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
                  className={styles.selectField}
                  filterOption={(input, option) =>
                    option.children.props.children[1].toLowerCase().includes(input.toLowerCase())
                  }
                  dropdownRender={(menu) => (
                    <div>
                      {menu}
                      <Divider style={{ margin: '8px 0' }} />
                      <div style={{ padding: '8px' }}>
                        <Button
                          type="text"
                          icon={<RobotOutlined />}
                          onClick={() => setShowAIRecommendation(true)}
                          style={{ 
                            width: '100%',
                            color: '#1890ff',
                            fontWeight: 500
                          }}
                        >
                          🤖 AI Gợi ý phân công thông minh
                        </Button>
                      </div>
                    </div>
                  )}
                >
                  {allAssignees.map((collab, index) => {
                    const username = collab.login || collab.github_username;
                    const uniqueKey = username ? `${username}-${index}` : `unknown-${index}`;
                    
                    return (
                      <Option key={uniqueKey} value={username}>
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
                <Select placeholder="Chọn độ ưu tiên..." className={styles.selectField}>
                  <Option value="low" label="Thấp">
                    <Space>
                      <FlagOutlined style={{ color: '#52c41a' }} />
                      <Text>Thấp</Text>
                    </Space>
                  </Option>
                  <Option value="medium" label="Trung bình">
                    <Space>
                      <FlagOutlined style={{ color: '#fa8c16' }} />
                      <Text>Trung bình</Text>
                    </Space>
                  </Option>
                  <Option value="high" label="Cao">
                    <Space>
                      <FlagOutlined style={{ color: '#f5222d' }} />
                      <Text>Cao</Text>
                    </Space>
                  </Option>
                  <Option value="urgent" label="Khẩn cấp">
                    <Space>
                      <FlagOutlined style={{ color: '#ff4d4f' }} />
                      <Text>Khẩn cấp</Text>
                    </Space>
                  </Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </div>

        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <CalendarOutlined className={styles.sectionIcon} />
            <Text strong>Thời gian</Text>
          </div>
          <Form.Item
            name="dueDate"
            label="Hạn hoàn thành"
          >
            <DatePicker
              style={{ width: '100%' }}
              placeholder="Chọn ngày hết hạn..."
              className={styles.inputField}
            />
          </Form.Item>
        </div>

        <div className={styles.modalFooter}>
          <Button
            size="large"
            onClick={() => setIsModalVisible(false)}
            className={styles.cancelButton}
          >
            Hủy bỏ
          </Button>
          <Button
            type="primary"
            htmlType="submit"
            size="large"
            className={styles.submitButton}
          >
            {editingTask ? '💾 Cập nhật' : '🚀 Tạo Task'}
          </Button>
        </div>
      </Form>

      {/* AI Assignment Recommendation Modal */}
      {selectedRepo && (
        <AssignmentRecommendation
          selectedRepo={selectedRepo}
          isVisible={showAIRecommendation}
          onClose={() => setShowAIRecommendation(false)}
          onSelectAssignee={handleAIAssigneeSelect}
          currentTaskData={getCurrentTaskData()}
        />
      )}
    </Modal>
  );
};

export default TaskModal;
