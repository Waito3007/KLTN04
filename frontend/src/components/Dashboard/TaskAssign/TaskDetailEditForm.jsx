import React, { useState, useEffect } from 'react';
import { Form, Input, Select, Row, Col, DatePicker, Spin } from 'antd';
import { UserOutlined } from '@ant-design/icons';

const { TextArea } = Input;
const { Option } = Select;

// Constants - đồng bộ với CreateTaskModal
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

const TaskDetailEditForm = ({ form, validationRules, selectedRepo }) => {
  const [collaborators, setCollaborators] = useState([]);
  const [loadingCollaborators, setLoadingCollaborators] = useState(false);

  useEffect(() => {
    if (selectedRepo) {
      fetchCollaborators(selectedRepo.owner, selectedRepo.name);
    }
  }, [selectedRepo]);

  const fetchCollaborators = async (owner, repo) => {
    setLoadingCollaborators(true);
    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
      const token = localStorage.getItem('access_token');
      const response = await fetch(`${apiBaseUrl}/repos/${owner}/${repo}/collaborators`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      const data = await response.json();
      setCollaborators(data);
    } catch (error) {
      console.error('Error fetching collaborators:', error);
    } finally {
      setLoadingCollaborators(false);
    }
  };

  return (
    <Form form={form} layout="vertical" name="editTaskForm">
      <Form.Item name="title" label="Tiêu đề" rules={validationRules.title}>
        <Input placeholder="Nhập tiêu đề task" showCount maxLength={255} />
      </Form.Item>

      <Row gutter={16}>
        <Col span={12}>
          <Form.Item name="status" label="Trạng thái">
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
          <Form.Item name="priority" label="Độ ưu tiên">
            <Select placeholder="Chọn độ ưu tiên">
              {TASK_PRIORITIES.map(priority => (
                <Option key={priority.value} value={priority.value}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        backgroundColor: priority.color,
                        marginRight: 8
                      }}
                    />
                    {priority.label}
                  </div>
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
            {loadingCollaborators ? (
              <Spin />
            ) : (
              <Select placeholder="Chọn người thực hiện">
                {collaborators.map(collab => (
                  <Option key={collab.login} value={collab.login}>
                    {collab.login}
                  </Option>
                ))}
              </Select>
            )}
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
  );
};

export default TaskDetailEditForm;
