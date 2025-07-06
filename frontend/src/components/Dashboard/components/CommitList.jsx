import React from 'react';
import { Card, List, Tag, Select, Pagination, Typography } from 'antd';
import { CodeOutlined, BugOutlined, ToolOutlined, FileTextOutlined } from '@ant-design/icons';

const { Text } = Typography;

/**
 * Component hiển thị danh sách commits
 */
const CommitList = ({ 
  memberCommits, 
  commitTypeFilter, 
  setCommitTypeFilter, 
  techAreaFilter, 
  setTechAreaFilter,
  currentPage,
  setCurrentPage,
  pageSize = 5
}) => {
  // Nếu không có dữ liệu commit, không hiển thị gì
  if (!memberCommits || !memberCommits.commits) return null;

  // Chuẩn hóa dữ liệu commit: hỗ trợ cả trường hợp API trả về trong analysis hoặc trực tiếp
  const rawCommits = memberCommits.commits || [];
  // Ưu tiên lấy stats trực tiếp nếu có, nếu không thì lấy các trường cũ
  const normalizeCommit = (commit) => {
    // Ưu tiên trường trực tiếp (backend mới), fallback sang stats hoặc các trường cũ
    const insertions = commit.insertions ?? commit.lines_added ?? commit.stats?.insertions ?? 0;
    const deletions = commit.deletions ?? commit.lines_removed ?? commit.stats?.deletions ?? 0;
    const files_changed = commit.files_changed ?? commit.files_count ?? commit.stats?.files_changed ?? 0;
    return {
      ...commit,
      insertions,
      deletions,
      files_changed,
    };
  };
  const normalizedCommits = rawCommits.map(normalizeCommit);

  // Filtered commits dựa trên filter
  const filteredCommits = normalizedCommits.filter(commit => {
    const commitType = commit.analysis?.type || commit.predicted_type || commit.type;
    const techArea = commit.analysis?.tech_area || commit.tech_area || 'general';
    const typeMatch = commitTypeFilter === 'all' || commitType === commitTypeFilter;
    const techMatch = techAreaFilter === 'all' || techArea === techAreaFilter;
    return typeMatch && techMatch;
  });

  // Phân trang
  const paginatedCommits = filteredCommits.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  // Unique commit types and tech areas for filter dropdowns
  const commitTypes = memberCommits.statistics ? 
    Object.keys(memberCommits.statistics.commit_types) : 
    memberCommits.multifusion_insights ? 
    Object.keys(memberCommits.multifusion_insights.commit_type_distribution) : [];
  
  const techAreas = memberCommits.statistics ? 
    Object.keys(memberCommits.statistics.tech_analysis || {}) : 
    ['general']; // MultiFusion doesn't have tech areas

  const getCommitTypeIcon = (type) => {
    const icons = {
      'feat': <CodeOutlined style={{ color: '#52c41a' }} />,
      'fix': <BugOutlined style={{ color: '#f5222d' }} />,
      'chore': <ToolOutlined style={{ color: '#1890ff' }} />,
      'docs': <FileTextOutlined style={{ color: '#fa8c16' }} />,
      'refactor': '♻️',
      'test': '✅',
      'style': '💄',
      'other': '📝'
    };
    return icons[type] || '📝';
  };

  const getCommitTypeColor = (type) => {
    const colors = {
      'feat': 'green',
      'fix': 'red',
      'chore': 'blue',
      'docs': 'orange',
      'refactor': 'purple',
      'test': 'cyan',
      'style': 'magenta',
      'other': 'default'
    };
    return colors[type] || 'default';
  };

  return (
    <Card title="📝 Danh sách commit gần đây" style={{ marginTop: '20px' }}>
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <Select
          value={commitTypeFilter}
          onChange={val => { setCommitTypeFilter(val); setCurrentPage(1); }}
          style={{ minWidth: 120 }}
        >
          <Select.Option value="all">Tất cả loại</Select.Option>
          {commitTypes.map(type => (
            <Select.Option key={type} value={type}>{type}</Select.Option>
          ))}
        </Select>
        <Select
          value={techAreaFilter}
          onChange={val => { setTechAreaFilter(val); setCurrentPage(1); }}
          style={{ minWidth: 120 }}
        >
          <Select.Option value="all">Tất cả lĩnh vực</Select.Option>
          {techAreas.map(area => (
            <Select.Option key={area} value={area}>{area}</Select.Option>
          ))}
        </Select>
      </div>
      <List
        dataSource={paginatedCommits}
        renderItem={commit => {
          // Handle both HAN and MultiFusion format
          const commitType = commit.analysis?.type || commit.predicted_type || commit.type || 'unknown';
          const techArea = commit.analysis?.tech_area || commit.tech_area || 'general';
          const confidence = commit.confidence || commit.analysis?.confidence || null;
          const isMultiFusion = !!commit.predicted_type;
          // Việt hóa commit type
          const commitTypeVN = {
            'feat': 'Tính năng',
            'fix': 'Sửa lỗi',
            'chore': 'Tác vụ',
            'docs': 'Tài liệu',
            'refactor': 'Cải tiến',
            'test': 'Kiểm thử',
            'style': 'Định dạng',
            'other': 'Khác',
            'unknown': 'Không xác định'
          }[commitType] || commitType;
          return (
            <List.Item>
              <List.Item.Meta
                title={
                  <div>
                    <span style={{ marginRight: '8px' }}>
                      {commit.message && commit.message.length > 80 ? 
                        commit.message.substring(0, 80) + '...' : 
                        commit.message || ''
                      }
                    </span>
                    <Tag 
                      color={getCommitTypeColor(commitType)}
                      icon={getCommitTypeIcon(commitType)}
                    >
                      {commit.analysis?.type_icon || ''} {commitTypeVN}
                    </Tag>
                    {!isMultiFusion && techArea !== 'general' && (
                      <Tag color="blue">{techArea}</Tag>
                    )}
                    {isMultiFusion && (
                      <Tag color="purple" style={{ fontSize: '10px' }}>
                        🔬 MultiFusion V2
                      </Tag>
                    )}
                    {confidence && (
                      <Tag color={confidence > 0.9 ? 'green' : confidence > 0.7 ? 'orange' : 'red'}>
                        {(confidence * 100).toFixed(1)}% độ tin cậy
                      </Tag>
                    )}
                    {commit.analysis?.ai_powered && !isMultiFusion && (
                      <>
                        {commit.analysis.impact && (
                          <Tag color={commit.analysis.impact === 'high' ? 'red' : 
                                    commit.analysis.impact === 'medium' ? 'orange' : 'green'}>
                            Tác động: {commit.analysis.impact === 'high' ? 'Cao' : 
                                      commit.analysis.impact === 'medium' ? 'Trung bình' : 'Thấp'}
                          </Tag>
                        )}
                        {commit.analysis.urgency && (
                          <Tag color={commit.analysis.urgency === 'urgent' ? 'red' : 
                                    commit.analysis.urgency === 'high' ? 'orange' : 'default'}>
                            {commit.analysis.urgency === 'urgent' ? 'Khẩn cấp' : 
                            commit.analysis.urgency === 'high' ? 'Cao' : commit.analysis.urgency}
                          </Tag>
                        )}
                        <Tag color="green" style={{ fontSize: '10px' }}>
                          🧠 HAN AI
                        </Tag>
                      </>
                    )}
                  </div>
                }
                description={
                  <div>
                    <Text code>{commit.sha || commit.commit_id || ''}</Text> •
                    <Text type="secondary">
                      {commit.date ? 
                        new Date(commit.date).toLocaleDateString('vi-VN') : 
                        'Ngày không xác định'
                      }
                    </Text>
                    {/* Thêm/xóa dòng và số file thay đổi */}
                    <Text style={{ color: '#52c41a', marginLeft: 8 }}>
                      +{commit.insertions} dòng thêm
                    </Text>
                    <Text style={{ color: '#f5222d', marginLeft: 8 }}>
                      -{commit.deletions} dòng xóa
                    </Text>
                    {commit.files_changed > 0 && (
                      <Text type="secondary" style={{ marginLeft: 8 }}>
                        {commit.files_changed} file thay đổi
                      </Text>
                    )}
                    {isMultiFusion && commit.detected_language && (
                      <Text type="secondary" style={{ marginLeft: 8 }}>
                        Ngôn ngữ: {commit.detected_language}
                      </Text>
                    )}
                  </div>
                }
              />
            </List.Item>
          );
        }}
      />
      <Pagination
        current={currentPage}
        pageSize={pageSize}
        total={filteredCommits.length}
        onChange={setCurrentPage}
        style={{ marginTop: 16, textAlign: 'center' }}
        showSizeChanger={false}
      />
    </Card>
  );
};

export default CommitList;
