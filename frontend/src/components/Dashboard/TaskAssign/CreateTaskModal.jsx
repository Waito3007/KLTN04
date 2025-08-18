/**
 * CreateTaskModal - Modal tạo task mới
 * Tuân thủ quy tắc KLTN04: Validation, defensive programming
 */

import React, { useState, useEffect } from 'react';
import { 
  Modal, 
  Form, 
  Input, 
  Select, 
  DatePicker, 
  Row, 
  Col, 
  Space, 
  Typography,
  Alert,
  Spin 
} from 'antd';
import { UserOutlined, CalendarOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import axios from 'axios'; // Thêm axios để gọi API

const { TextArea } = Input;
const { Option } = Select;
const { Text } = Typography;

// Constants - tuân thủ nguyên tắc không magic values
const TASK_PRIORITIES = [
  { value: 'LOW', label: 'Thấp', color: '#52c41a' },
  { value: 'MEDIUM', label: 'Trung bình', color: '#1890ff' },
  { value: 'HIGH', label: 'Cao', color: '#fa8c16' },
  { value: 'URGENT', label: 'Khẩn cấp', color: '#ff4d4f' }
];

const TASK_STATUSES = [
  { value: 'TODO', label: 'Cần làm' },
  { value: 'IN_PROGRESS', label: 'Đang làm' },
  { value: 'DONE', label: 'Hoàn thành' }
];

const CreateTaskModal = ({ 
  visible, 
  onCancel, 
  onSubmit, 
  selectedRepo, 
  loading = false 
}) => {
  const [form] = Form.useForm();
  const [submitLoading, setSubmitLoading] = useState(false);
  const [collaborators, setCollaborators] = useState([]); // State lưu danh sách thành viên
  const [loadingCollaborators, setLoadingCollaborators] = useState(false); // State loading

  // Reset form khi modal mở/đóng
  useEffect(() => {
    if (visible) {
      form.resetFields();
      form.setFieldsValue({
        status: 'TODO',
        priority: 'MEDIUM',
      });

      // Load danh sách thành viên trong repo
      if (selectedRepo) {
        fetchCollaborators(selectedRepo.owner, selectedRepo.name);
      }
    }
  }, [visible, form, selectedRepo]);

  // Hàm gọi API lấy danh sách thành viên
  const fetchCollaborators = async (owner, repo) => {
    setLoadingCollaborators(true);
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL; // Lấy URL từ biến môi trường
      const token = localStorage.getItem('access_token'); // Lấy token từ localStorage
      const ownerString = typeof owner === 'string' ? owner : owner?.login || 'unknown';

      // Gọi API lấy danh sách cộng tác viên từ database
      const response = await axios.get(`${apiBaseUrl}/github/${ownerString}/${repo}/collaborators`, {
        headers: {
          'Authorization': `Bearer ${token}`, // Gửi token trong header
          'Cache-Control': 'no-cache',
        },
      });

      if (response.data.collaborators && response.data.collaborators.length > 0) {
        setCollaborators(response.data.collaborators);
      } else {
        // Nếu không có cộng tác viên, hiển thị nút để đồng bộ thủ công
        setCollaborators([]);
      }
    } catch (error) {
      console.error('Lỗi khi lấy danh sách thành viên từ database:', error);
      setCollaborators([]);
    } finally {
      setLoadingCollaborators(false);
    }
  };

  const syncCollaborators = async (owner, repo) => {
    setLoadingCollaborators(true);
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL; // Lấy URL từ biến môi trường
      const token = localStorage.getItem('access_token'); // Lấy token từ localStorage
      const ownerString = typeof owner === 'string' ? owner : owner?.login || 'unknown';

      // Gọi API đồng bộ cộng tác viên
      const response = await axios.post(`${apiBaseUrl}/github/${ownerString}/${repo}/collaborators/sync`, {}, {
        headers: {
          'Authorization': `Bearer ${token}`, // Gửi token trong header
          'Cache-Control': 'no-cache',
        },
      });

      setCollaborators(response.data.collaborators || []);
    } catch (error) {
      console.error('Lỗi khi đồng bộ danh sách thành viên:', error);
      setCollaborators([]);
    } finally {
      setLoadingCollaborators(false);
    }
  };

  // Validation rules - defensive programming
  const validationRules = {
    title: [
      { required: true, message: 'Vui lòng nhập tiêu đề task' },
      { min: 3, message: 'Tiêu đề phải có ít nhất 3 ký tự' },
      { max: 255, message: 'Tiêu đề không được quá 255 ký tự' }
    ],
    description: [
      { max: 1000, message: 'Mô tả không được quá 1000 ký tự' }
    ],
    assignee: [
      { pattern: /^[a-zA-Z0-9]([a-zA-Z0-9-])*[a-zA-Z0-9]$/, message: 'GitHub username không hợp lệ' }
    ],
    due_date: [
      {
        validator: (_, value) => {
          if (!value) return Promise.resolve();
          if (dayjs(value).isBefore(dayjs(), 'day')) {
            return Promise.reject('Ngày hết hạn không được ở quá khứ');
          }
          return Promise.resolve();
        }
      }
    ]
  };

  // Handle submit với error handling
  const handleSubmit = async () => {
    try {
      setSubmitLoading(true);
      
      // Validate form
      const values = await form.validateFields();
      
      // Defensive programming: Validate dữ liệu trước khi submit
      if (!values.title?.trim()) {
        throw new Error('Tiêu đề task không được để trống');
      }

      // Format data
      const taskData = {
        title: values.title.trim(),
        description: values.description?.trim() || '',
        status: values.status || 'TODO',
        priority: values.priority || 'MEDIUM',
        assignee_github_username: values.assignee_github_username?.trim() || null,
        due_date: values.due_date ? dayjs(values.due_date).format('YYYY-MM-DD') : null
      };

      await onSubmit?.(taskData);
      
    } catch (error) {
      console.error('Lỗi khi tạo task:', error);
      // Error được handle bởi parent component
    } finally {
      setSubmitLoading(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel?.();
  };

  return (
    <Modal
      title="Tạo Task Mới"
      open={visible}
      onCancel={handleCancel}
      onOk={handleSubmit}
      confirmLoading={submitLoading || loading}
      width={600}
      destroyOnClose
      okText="Tạo Task"
      cancelText="Hủy"
    >
      {/* Repository Info */}
      {selectedRepo && (
        <Alert
          message={`Repository: ${typeof selectedRepo.owner === 'string' ? selectedRepo.owner : selectedRepo.owner?.login || 'unknown'}/${selectedRepo.name}`}
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Form
        form={form}
        layout="vertical"
        name="createTaskForm"
        preserve={false}
      >
        {/* Task Title */}
        <Form.Item
          name="title"
          label="Tiêu đề Task"
          rules={validationRules.title}
        >
          <Input 
            placeholder="Nhập tiêu đề cho task"
            showCount
            maxLength={255}
          />
        </Form.Item>

        {/* Task Description */}
        <Form.Item
          name="description"
          label="Mô tả"
          rules={validationRules.description}
        >
          <TextArea
            placeholder="Nhập mô tả chi tiết cho task (tùy chọn)"
            rows={4}
            showCount
            maxLength={1000}
          />
        </Form.Item>

        {/* Status and Priority */}
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="status"
              label="Trạng thái"
              initialValue="TODO"
            >
              <Select placeholder="Chọn trạng thái">
                {TASK_STATUSES.map(status => (
                  <Option key={status.value} value={status.value}>
                    {status.label}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
          
          <Col span={12}>
            <Form.Item
              name="priority"
              label="Độ ưu tiên"
              initialValue="MEDIUM"
            >
              <Select placeholder="Chọn độ ưu tiên">
                {TASK_PRIORITIES.map(priority => (
                  <Option key={priority.value} value={priority.value}>
                    <Space>
                      <div 
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: priority.color
                        }}
                      />
                      {priority.label}
                    </Space>
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>

        {/* Assignee and Due Date */}
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              name="assignee_github_username"
              label="Người thực hiện"
            >
              {loadingCollaborators ? (
                <Spin />
              ) : (
                <Select placeholder="Chọn người thực hiện">
                  {Array.isArray(collaborators) && collaborators.map((collab) => (
                    <Option key={collab.login} value={collab.login}>
                      {collab.login}
                    </Option>
                  ))}
                </Select>
              )}
            </Form.Item>
          </Col>
          
          <Col span={12}>
            <Form.Item
              name="due_date"
              label="Ngày hết hạn"
              rules={validationRules.due_date}
            >
              <DatePicker
                style={{ width: '100%' }}
                placeholder="Chọn ngày hết hạn"
                format="DD/MM/YYYY"
                disabledDate={(current) => current && current.isBefore(dayjs(), 'day')}
                suffixIcon={<CalendarOutlined />}
              />
            </Form.Item>
          </Col>
        </Row>

        {/* Nếu không có cộng tác viên, hiển thị nút cập nhật */}
        {Array.isArray(collaborators) && collaborators.length === 0 && !loadingCollaborators && (
          <Alert
            message="Không tìm thấy cộng tác viên nào."
            description={
              <Space>
                <span>Vui lòng nhấn nút bên dưới để đồng bộ danh sách cộng tác viên từ GitHub.</span>
                <button
                  onClick={() => syncCollaborators(selectedRepo.owner, selectedRepo.name)}
                  style={{ backgroundColor: '#1890ff', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer' }}
                >
                  Đồng bộ cộng tác viên
                </button>
              </Space>
            }
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* Nút đồng bộ chủ động */}
        {Array.isArray(collaborators) && collaborators.length > 0 && (
          <div style={{ marginBottom: 16, textAlign: 'right' }}>
            <button
              onClick={() => syncCollaborators(selectedRepo.owner, selectedRepo.name)}
              style={{ backgroundColor: '#1890ff', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer' }}
            >
              Đồng bộ cộng tác viên
            </button>
          </div>
        )}

        {/* Helper Text */}
        <div style={{ marginTop: 16 }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            * Task sẽ được tạo trong repository {typeof selectedRepo?.owner === 'string' ? selectedRepo.owner : selectedRepo?.owner?.login || 'unknown'}/{selectedRepo?.name}
          </Text>
        </div>
      </Form>
    </Modal>
  );
};

export default CreateTaskModal;
