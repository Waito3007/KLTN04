import React from 'react';
import { Modal, Form, Input, Select, DatePicker, Button, Space, Avatar, Tag } from 'antd';

const { Option } = Select;
const { TextArea } = Input;

const TaskModal = ({
  isModalVisible,
  editingTask,
  form,
  handleTaskSubmit,
  setIsModalVisible,
  collaborators
}) => (
  <Modal
    title={editingTask ? "Chỉnh sửa Task" : "Tạo Task Mới"}
    open={isModalVisible}
    onCancel={() => setIsModalVisible(false)}
    footer={null}
  >
    <Form
      form={form}
      layout="vertical"
      onFinish={handleTaskSubmit}
    >
      <Form.Item
        name="title"
        label="Tiêu đề"
        rules={[{ required: true, message: 'Vui lòng nhập tiêu đề!' }]}
      >
        <Input placeholder="Nhập tiêu đề task" />
      </Form.Item>
      <Form.Item
        name="description"
        label="Mô tả"
        rules={[{ required: true, message: 'Vui lòng nhập mô tả!' }]}
      >
        <TextArea rows={3} placeholder="Mô tả chi tiết task" />
      </Form.Item>
      <Form.Item
        name="assignee"
        label="Giao cho"
        rules={[{ required: true, message: 'Vui lòng chọn người thực hiện!' }]}
      >
        <Select 
          placeholder="Chọn thành viên"
          showSearch
          optionFilterProp="children"
          filterOption={(input, option) =>
            option.children.props.children[1].toLowerCase().indexOf(input.toLowerCase()) >= 0
          }
        >
          {collaborators.map(collab => (
            <Option key={collab.login} value={collab.login}>
              <Space>
                <Avatar src={collab.avatar_url} size="small" />
                <span>{collab.login}</span>
                {collab.type === 'Owner' && <Tag color="gold">Owner</Tag>}
                {collab.type === 'Contributor' && <Tag color="blue">Contributor</Tag>}
                {collab.contributions > 0 && (
                  <Tag color="green" style={{ fontSize: '10px' }}>
                    {collab.contributions} commits
                  </Tag>
                )}
              </Space>
            </Option>
          ))}
        </Select>
      </Form.Item>
      <Form.Item
        name="priority"
        label="Độ ưu tiên"
        rules={[{ required: true, message: 'Vui lòng chọn độ ưu tiên!' }]}
      >
        <Select placeholder="Chọn độ ưu tiên">
          <Option value="low">
            <Tag color="#52c41a">Thấp</Tag>
          </Option>
          <Option value="medium">
            <Tag color="#fa8c16">Trung bình</Tag>
          </Option>
          <Option value="high">
            <Tag color="#f5222d">Cao</Tag>
          </Option>
        </Select>
      </Form.Item>
      <Form.Item
        name="dueDate"
        label="Hạn hoàn thành"
      >
        <DatePicker style={{ width: '100%' }} />
      </Form.Item>
      <Form.Item>
        <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
          <Button onClick={() => setIsModalVisible(false)}>
            Hủy
          </Button>
          <Button type="primary" htmlType="submit">
            {editingTask ? 'Cập nhật' : 'Tạo mới'}
          </Button>
        </Space>
      </Form.Item>
    </Form>
  </Modal>
);

export default TaskModal;
