/**
 * TaskDetailModal - Modal hiển thị chi tiết và chỉnh sửa task
 * Tuân thủ quy tắc KLTN04: Defensive programming, validation
 */

import React, { useState, useEffect } from 'react';
import { Modal, Button, Space, Popconfirm, Row } from 'antd';
import { EditOutlined, DeleteOutlined } from '@ant-design/icons';
import { Form } from 'antd';
import dayjs from 'dayjs';
import { parseISO, format } from 'date-fns';
import { vi } from 'date-fns/locale';
import TaskDetailView from './TaskDetailView';
import TaskDetailEditForm from './TaskDetailEditForm';
import TaskDetailCommits from './TaskDetailCommits';
import TaskCommitLinker from './TaskCommitLinker';

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
  const [relatedCommits, setRelatedCommits] = useState([]);
  const [loadingCommits, setLoadingCommits] = useState(false);
  const [showCommitLinker, setShowCommitLinker] = useState(false);

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
      setRelatedCommits([]);
    }
  }, [visible, task, form]);

  // Fetch commits liên quan đến task
  const fetchRelatedCommits = async (searchTerm = '') => {
    if (!task) return;
    setLoadingCommits(true);
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
      const token = localStorage.getItem('access_token');
      const response = await fetch(
        `${apiBaseUrl}/tasks/${task.id}/commits?search=${encodeURIComponent(searchTerm)}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await response.json();
      setRelatedCommits(data || []);
    } catch (error) {
      console.error('Lỗi khi tìm kiếm commits:', error);
    } finally {
      setLoadingCommits(false);
    }
  };

  // Liên kết commits với task
  const linkCommitsToTask = async () => {
    if (!task) return;
    setLoadingCommits(true);
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
      const token = localStorage.getItem('access_token');
      const response = await fetch(
        `${apiBaseUrl}/tasks/${task.id}/link-commits`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await response.json();
      if (data.success) {
        setRelatedCommits(data.commits || []);
      }
    } catch (error) {
      console.error('Lỗi khi liên kết commits:', error);
    } finally {
      setLoadingCommits(false);
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

  return (
    <Modal
      title={
        <Space>
          <span>Chi tiết Task</span>
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
          <Space key="actions" style={{ width: '100%', justifyContent: 'space-between' }}>
            <Space>
              <Button
                onClick={() => fetchRelatedCommits()}
                loading={loadingCommits}
              >
                Tìm Commits
              </Button>
              <Button
                onClick={() => setShowCommitLinker(true)}
                type="primary"
                disabled={!task?.assignee_github_username}
              >
                Liên kết Commits
              </Button>
            </Space>
            <Space>
              <Popconfirm
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
              </Popconfirm>
              <Button onClick={handleCancel}>
                Đóng
              </Button>
            </Space>
          </Space>
        ]
      }
    >
      {isEditing ? (
        <TaskDetailEditForm
          form={form}
          validationRules={{
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
          }}
          task={task}
        />
      ) : (
        <TaskDetailView
          task={task}
          selectedRepo={selectedRepo}
          formatDate={(dateString) => {
            if (!dateString) return 'Không có';
            try {
              const date = parseISO(dateString);
              return format(date, 'dd/MM/yyyy HH:mm', { locale: vi });
            } catch {
              return 'Không hợp lệ';
            }
          }}
          formatDueDate={(dateString) => {
            if (!dateString) return 'Không có';
            try {
              const date = parseISO(dateString);
              return format(date, 'dd/MM/yyyy', { locale: vi });
            } catch {
              return 'Không hợp lệ';
            }
          }}
          getPriorityConfig={(priority) => {
            return TASK_PRIORITIES.find(p => p.value === priority) || TASK_PRIORITIES[1];
          }}
          getStatusConfig={(status) => {
            return TASK_STATUSES.find(s => s.value === status) || TASK_STATUSES[0];
          }}
        />
      )}

      <TaskDetailCommits
        relatedCommits={relatedCommits}
        loadingCommits={loadingCommits}
        fetchRelatedCommits={fetchRelatedCommits}
        linkCommitsToTask={linkCommitsToTask}
        formatDate={(dateString) => {
          if (!dateString) return 'Không có';
          try {
            const date = parseISO(dateString);
            return format(date, 'dd/MM/yyyy HH:mm', { locale: vi });
          } catch {
            return 'Không hợp lệ';
          }
        }}
      />

      {/* Task Commit Linker */}
      <TaskCommitLinker
        task={task}
        selectedRepo={selectedRepo}
        visible={showCommitLinker}
        onClose={() => setShowCommitLinker(false)}
      />
    </Modal>
  );
};

export default TaskDetailModal;
