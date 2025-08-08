/**
 * TaskDetailModal - Modal hiển thị chi tiết và chỉnh sửa task
 * Tuân thủ quy tắc KLTN04: Defensive programming, validation
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
  Tag,
  Button,
  Popconfirm,
  Divider,
  Alert
} from 'antd';
import {
  EditOutlined,
  DeleteOutlined,
  CalendarOutlined,
  UserOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { format, parseISO } from 'date-fns';
import { vi } from 'date-fns/locale';

const { TextArea } = Input;
const { Option } = Select;
const { Text, Title } = Typography;

// Constants
const TASK_PRIORITIES = [
  { value: 'LOW', label: 'Thấp', color: '#52c41a' },
  { value: 'MEDIUM', label: 'Trung bình', color: '#1890ff' },
  { value: 'HIGH', label: 'Cao', color: '#fa8c16' },
  { value: 'URGENT', label: 'Khẩn cấp', color: '#ff4d4f' }
];

const TASK_STATUSES = [
  { value: 'TODO', label: 'Cần làm', color: '#1890ff' },
  { value: 'IN_PROGRESS', label: 'Đang làm', color: '#fa8c16' },
  { value: 'DONE', label: 'Hoàn thành', color: '#52c41a' },
  { value: 'CANCELLED', label: 'Đã hủy', color: '#ff4d4f' }
];

const TaskDetailModal = ({
  visible,
  task,
  onCancel,
  onUpdate,
  onDelete,
  selectedRepo
}) => {
  const [form] = Form.useForm();
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);

  // Reset form khi task thay đổi
  useEffect(() => {
    if (visible && task) {
      form.setFieldsValue({
        title: task.title,
        description: task.description,
        status: task.status,
        priority: task.priority,
        assignee_github_username: task.assignee_github_username,
        due_date: task.due_date ? dayjs(task.due_date) : null
      });
      setIsEditing(false);
    }
  }, [visible, task, form]);

  // Validation rules
  const validationRules = {
    title: [
      { required: true, message: 'Vui lòng nhập tiêu đề task' },
      { min: 3, message: 'Tiêu đề phải có ít nhất 3 ký tự' },
      { max: 255, message: 'Tiêu đề không được quá 255 ký tự' }
    ],
    description: [
      { max: 1000, message: 'Mô tả không được quá 1000 ký tự' }
    ],
    assignee_github_username: [
      { pattern: /^[a-zA-Z0-9]([a-zA-Z0-9-])*[a-zA-Z0-9]$/, message: 'GitHub username không hợp lệ' }
    ]
  };

  // Get config cho priority và status
  const getPriorityConfig = (priority) => {
    return TASK_PRIORITIES.find(p => p.value === priority) || TASK_PRIORITIES[1];
  };

  const getStatusConfig = (status) => {
    return TASK_STATUSES.find(s => s.value === status) || TASK_STATUSES[0];
  };

  // Format dates
  const formatDate = (dateString) => {
    if (!dateString) return 'Không có';
    try {
      const date = parseISO(dateString);
      return format(date, 'dd/MM/yyyy HH:mm', { locale: vi });
    } catch {
      return 'Không hợp lệ';
    }
  };

  const formatDueDate = (dateString) => {
    if (!dateString) return 'Không có';
    try {
      const date = parseISO(dateString);
      return format(date, 'dd/MM/yyyy', { locale: vi });
    } catch {
      return 'Không hợp lệ';
    }
  };

  // Handlers
  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    // Reset form về giá trị ban đầu
    if (task) {
      form.setFieldsValue({
        title: task.title,
        description: task.description,
        status: task.status,
        priority: task.priority,
        assignee_github_username: task.assignee_github_username,
        due_date: task.due_date ? dayjs(task.due_date) : null
      });
    }
  };

  const handleUpdate = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();
      
      const updateData = {
        title: values.title?.trim(),
        description: values.description?.trim() || '',
        status: values.status,
        priority: values.priority,
        assignee_github_username: values.assignee_github_username?.trim() || null,
        due_date: values.due_date ? dayjs(values.due_date).format('YYYY-MM-DD') : null
      };

      await onUpdate?.(task.id, updateData);
      setIsEditing(false);
    } catch (error) {
      console.error('Lỗi khi cập nhật task:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    try {
      setLoading(true);
      await onDelete?.(task.id);
    } catch (error) {
      console.error('Lỗi khi xóa task:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    onCancel?.();
  };

  if (!task) return null;

  const priorityConfig = getPriorityConfig(task.priority);
  const statusConfig = getStatusConfig(task.status);

  return (
    <Modal
      title={
        <Space>
          <Text>Chi tiết Task</Text>
          {!isEditing && (
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={handleEdit}
            >
              Chỉnh sửa
            </Button>
          )}
        </Space>
      }
      open={visible}
      onCancel={handleCancel}
      width={700}
      footer={
        isEditing ? [
          <Button key="cancel" onClick={handleCancelEdit}>
            Hủy
          </Button>,
          <Button
            key="save"
            type="primary"
            loading={loading}
            onClick={handleUpdate}
          >
            Lưu thay đổi
          </Button>
        ] : [
          <Popconfirm
            key="delete"
            title="Bạn có chắc chắn muốn xóa task này?"
            description="Hành động này không thể hoàn tác."
            onConfirm={handleDelete}
            okText="Xóa"
            cancelText="Hủy"
            okType="danger"
          >
            <Button
              danger
              icon={<DeleteOutlined />}
              loading={loading}
            >
              Xóa Task
            </Button>
          </Popconfirm>,
          <Button key="close" onClick={handleCancel}>
            Đóng
          </Button>
        ]
      }
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

      {isEditing ? (
        // Edit Mode
        <Form form={form} layout="vertical" name="editTaskForm">
          <Form.Item name="title" label="Tiêu đề" rules={validationRules.title}>
            <Input placeholder="Nhập tiêu đề task" showCount maxLength={255} />
          </Form.Item>

          <Form.Item name="description" label="Mô tả" rules={validationRules.description}>
            <TextArea
              placeholder="Nhập mô tả chi tiết"
              rows={4}
              showCount
              maxLength={1000}
            />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="status" label="Trạng thái">
                <Select>
                  {TASK_STATUSES.map(status => (
                    <Option key={status.value} value={status.value}>
                      <Tag color={status.color}>{status.label}</Tag>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="priority" label="Độ ưu tiên">
                <Select>
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

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="assignee_github_username"
                label="Người thực hiện"
                rules={validationRules.assignee_github_username}
              >
                <Input placeholder="GitHub username" prefix={<UserOutlined />} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="due_date" label="Ngày hết hạn">
                <DatePicker
                  style={{ width: '100%' }}
                  format="DD/MM/YYYY"
                  placeholder="Chọn ngày hết hạn"
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      ) : (
        // View Mode
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          {/* Title */}
          <div>
            <Title level={4} style={{ margin: 0 }}>
              {task.title}
            </Title>
          </div>

          {/* Status and Priority */}
          <Space size="middle">
            <Tag color={statusConfig.color} style={{ fontSize: '14px', padding: '4px 12px' }}>
              {statusConfig.label}
            </Tag>
            <Tag color={priorityConfig.color} style={{ fontSize: '14px', padding: '4px 12px' }}>
              {priorityConfig.label}
            </Tag>
          </Space>

          {/* Description */}
          {task.description && (
            <div>
              <Text strong>Mô tả:</Text>
              <div style={{ marginTop: 8, padding: 12, backgroundColor: '#fafafa', borderRadius: 6 }}>
                <Text>{task.description}</Text>
              </div>
            </div>
          )}

          <Divider />

          {/* Task Details */}
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Space>
                <UserOutlined />
                <Text strong>Người thực hiện:</Text>
                <Text>{task.assignee_github_username || 'Chưa phân công'}</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Space>
                <CalendarOutlined />
                <Text strong>Hạn cuối:</Text>
                <Text>{formatDueDate(task.due_date)}</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Space>
                <ClockCircleOutlined />
                <Text strong>Tạo lúc:</Text>
                <Text>{formatDate(task.created_at)}</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Space>
                <ClockCircleOutlined />
                <Text strong>Cập nhật:</Text>
                <Text>{formatDate(task.updated_at)}</Text>
              </Space>
            </Col>
            {task.created_by && (
              <Col span={24}>
                <Space>
                  <UserOutlined />
                  <Text strong>Tạo bởi:</Text>
                  <Text>{task.created_by}</Text>
                </Space>
              </Col>
            )}
          </Row>
        </Space>
      )}
    </Modal>
  );
};

export default TaskDetailModal;
