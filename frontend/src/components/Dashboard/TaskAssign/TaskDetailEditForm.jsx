import React from 'react';
import { Form, Input, Select, Row, Col, DatePicker } from 'antd';
import { UserOutlined } from '@ant-design/icons';

const { TextArea } = Input;
const { Option } = Select;

const TaskDetailEditForm = ({ form, validationRules, task }) => {
  return (
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
              {task.statuses.map(status => (
                <Option key={status.value} value={status.value}>
                  {status.label}
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item name="priority" label="Độ ưu tiên">
            <Select>
              {task.priorities.map(priority => (
                <Option key={priority.value} value={priority.value}>
                  {priority.label}
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
  );
};

export default TaskDetailEditForm;
