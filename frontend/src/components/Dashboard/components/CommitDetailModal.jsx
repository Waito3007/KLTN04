import React from 'react';
import { Modal, Typography, Tag, List, Divider, Row, Col, Space } from 'antd';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const { Text, Title } = Typography;

const CommitDetailModal = ({ visible, onClose, commit }) => {
  if (!commit) return null;
  return (
    <Modal
      open={visible}
      onCancel={onClose}
      footer={null}
      width={Math.min(window.innerWidth - 32, 900)}
      title={<Title level={5}>Chi tiết commit</Title>}
      styles={{ body: { padding: 24 } }}
    >
      <Row gutter={[24, 24]}>
        <Col xs={24} md={16}>
          <Text strong style={{ fontSize: 16 }}>{commit.message}</Text>
          <div style={{ marginTop: 8, marginBottom: 8 }}>
            <Tag color="default">{commit.author_name}</Tag>
            {commit.branch_name && <Tag color="blue">{commit.branch_name}</Tag>}
            {commit.analysis?.type && <Tag color="geekblue">{commit.analysis.type}</Tag>}
            {commit.analysis?.ai_powered && <Tag color="green">AI</Tag>}
            {commit.analysis?.ai_model && <Tag color="purple">{commit.analysis.ai_model}</Tag>}
            {commit.sha_short && <Tag color="default">SHA: {commit.sha_short}</Tag>}
          </div>
          <Space wrap style={{ marginBottom: 8 }}>
            <Tag color="green">+{commit.insertions} dòng</Tag>
            <Tag color="red">-{commit.deletions} dòng</Tag>
            <Tag color="blue">{commit.files_changed} file</Tag>
            <Tag color="default">Ngày: {commit.date || commit.committer_date}</Tag>
          </Space>
        </Col>
        <Col xs={24} md={8}>
          <div style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
            <Text strong>Thông tin khác</Text>
            <List size="small" style={{ marginTop: 8 }}>
              <List.Item><Text type="secondary">SHA</Text>: <Text code>{commit.sha}</Text></List.Item>
              {commit.branch_name && <List.Item><Text type="secondary">Branch</Text>: <Text code>{commit.branch_name}</Text></List.Item>}
              <List.Item><Text type="secondary">Ngày</Text>: <Text>{commit.date || commit.committer_date}</Text></List.Item>
            </List>
          </div>
        </Col>
      </Row>
      <Divider />
      <Title level={5} style={{ marginBottom: 8 }}>Code diff</Title>
      <SyntaxHighlighter
        language={'diff'}
        style={vscDarkPlus}
        customStyle={{ borderRadius: 8, fontSize: 14, marginTop: 8 }}
        showLineNumbers
        wrapLines
      >
        {commit.diff_content || '// Không có diff cho commit này'}
      </SyntaxHighlighter>
    </Modal>
  );
};

export default CommitDetailModal;
