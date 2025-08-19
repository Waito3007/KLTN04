import React, { useState, useEffect, useCallback } from 'react';
import { Modal, Button, List, Checkbox, message, Card, Typography, Space, Tag, Spin, Input, Pagination, Select } from 'antd';
import { LinkOutlined, CodeOutlined, UserOutlined, CalendarOutlined, SearchOutlined, TeamOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Text, Title } = Typography;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const TaskCommitLinker = ({ task, selectedRepo, visible, onClose, onCommitLinked }) => {
  const [commits, setCommits] = useState([]);
  const [linkedCommits, setLinkedCommits] = useState([]);
  const [selectedCommitIds, setSelectedCommitIds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [linking, setLinking] = useState(false);
  
  // Pagination states
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCommits, setTotalCommits] = useState(0);
  const [pageSize] = useState(10);
  
  // Search states
  const [searchQuery, setSearchQuery] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);

  // Author selection states
  const [availableAuthors, setAvailableAuthors] = useState([]);
  const [selectedAuthor, setSelectedAuthor] = useState(task?.assignee_github_username);
  const [authorsLoading, setAuthorsLoading] = useState(false);

  // Lấy danh sách authors có trong repository
  const fetchAvailableAuthors = useCallback(async () => {
    if (!selectedRepo) return;
    
    setAuthorsLoading(true);
    try {
      const owner = selectedRepo.owner?.login || selectedRepo.owner;
      const repo = selectedRepo.name;
      
      const token = localStorage.getItem('access_token');
      
      const response = await axios.get(
        `${API_BASE_URL}/projects/${owner}/${repo}/authors`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );
      
      const authors = response.data.authors || [];
      setAvailableAuthors(authors);
      
      // Tự động chọn author phù hợp nhất
      if (authors.length > 0 && !selectedAuthor) {
        // Tìm author có commit_count cao nhất làm mặc định
        const bestAuthor = authors[0];
        setSelectedAuthor(bestAuthor.author_name);
      }
    } catch (error) {
      console.error('Error fetching authors:', error);
      message.error('Không thể lấy danh sách tác giả');
    } finally {
      setAuthorsLoading(false);
    }
  }, [selectedRepo, selectedAuthor]);

  // Lấy danh sách commit của người được chọn với phân trang và tìm kiếm
  const fetchUserCommits = useCallback(async (page = 1, search = '') => {
    if (!selectedAuthor || !selectedRepo) return;
    
    setLoading(true);
    try {
      const owner = selectedRepo.owner?.login || selectedRepo.owner;
      const repo = selectedRepo.name;
      
      const params = new URLSearchParams({
        limit: pageSize.toString(),
        offset: ((page - 1) * pageSize).toString()
      });
      
      if (search.trim()) {
        params.append('search', search.trim());
      }
      
      // Lấy token từ localStorage
      const token = localStorage.getItem('access_token');
      
      const response = await axios.get(
        `${API_BASE_URL}/projects/${owner}/${repo}/users/${selectedAuthor}/commits?${params}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );
      
      setCommits(response.data.commits || response.data);
      setTotalCommits(response.data.total || response.data.length);
      setCurrentPage(page);
    } catch (error) {
      console.error('Error fetching user commits:', error);
      message.error(`Không thể lấy danh sách commit của ${selectedAuthor}`);
    } finally {
      setLoading(false);
    }
  }, [selectedAuthor, selectedRepo, pageSize]);

  const fetchLinkedCommits = useCallback(async () => {
    if (!task?.id || !selectedRepo) return;

    try {
      const owner = selectedRepo.owner?.login || selectedRepo.owner;
      const repo = selectedRepo.name;
      
      // Lấy token từ localStorage
      const token = localStorage.getItem('access_token');
      
      const response = await axios.get(
        `${API_BASE_URL}/projects/${owner}/${repo}/tasks/${task.id}/commits`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );
      setLinkedCommits(response.data);
    } catch (error) {
      console.error('Error fetching linked commits:', error);
    }
  }, [task?.id, selectedRepo]);

  useEffect(() => {
    if (visible && selectedRepo) {
      fetchAvailableAuthors();
    }
  }, [visible, selectedRepo, fetchAvailableAuthors]);

  useEffect(() => {
    if (visible && selectedAuthor && selectedRepo) {
      fetchUserCommits(1, '');
      fetchLinkedCommits();
      setCurrentPage(1);
      setSearchQuery('');
      setSelectedCommitIds([]);
    }
  }, [visible, selectedAuthor, selectedRepo, fetchUserCommits, fetchLinkedCommits]);

  // Hàm xử lý thay đổi author
  const handleAuthorChange = useCallback((authorName) => {
    setSelectedAuthor(authorName);
    setCommits([]);
    setSelectedCommitIds([]);
    setCurrentPage(1);
    setSearchQuery('');
  }, []);

  // Hàm tìm kiếm
  const handleSearch = useCallback(async (value) => {
    setSearchLoading(true);
    setSearchQuery(value);
    await fetchUserCommits(1, value);
    setSearchLoading(false);
  }, [fetchUserCommits]);

  // Hàm thay đổi trang
  const handlePageChange = useCallback(async (page) => {
    await fetchUserCommits(page, searchQuery);
  }, [fetchUserCommits, searchQuery]);

  const handleCommitSelection = (commitSha, checked) => {
    if (checked) {
      setSelectedCommitIds([...selectedCommitIds, commitSha]);
    } else {
      setSelectedCommitIds(selectedCommitIds.filter(id => id !== commitSha));
    }
  };

  const handleLinkCommits = async () => {
    if (selectedCommitIds.length === 0) {
      message.warning('Vui lòng chọn ít nhất một commit để liên kết');
      return;
    }

    setLinking(true);
    try {
      const owner = selectedRepo.owner?.login || selectedRepo.owner;
      const repo = selectedRepo.name;

      // Lấy token từ localStorage
      const token = localStorage.getItem('access_token');

      // Gọi API để liên kết các commit đã chọn
      const response = await axios.post(
        `${API_BASE_URL}/projects/${owner}/${repo}/tasks/${task.id}/link-commits`,
        {
          commit_shas: selectedCommitIds
        },
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data.success) {
        message.success(`Đã liên kết ${selectedCommitIds.length} commit với task`);
        setSelectedCommitIds([]);
        await fetchLinkedCommits(); // Refresh danh sách commit đã liên kết
        if (onCommitLinked) onCommitLinked(); // Callback để refresh commits ở component cha
        onClose();
      }
    } catch (error) {
      console.error('Error linking commits:', error);
      message.error('Không thể liên kết commit với task');
    } finally {
      setLinking(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Không có';
    try {
      return new Date(dateString).toLocaleString('vi-VN');
    } catch {
      return 'Không hợp lệ';
    }
  };

  const isCommitLinked = (commitSha) => {
    return linkedCommits.some(linked => linked.sha === commitSha);
  };

  return (
    <Modal
      title={
        <Space>
          <LinkOutlined />
          <span>Liên kết Commit với Task: {task?.title}</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={800}
      footer={null}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* Thông tin task và chọn tác giả */}
        <Card size="small" style={{ backgroundColor: '#f8f9fa' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Space>
              <UserOutlined /> 
              <Text strong>Người được phân công: {task?.assignee_github_username || 'Chưa phân công'}</Text>
            </Space>
            <Space>
              <CodeOutlined /> 
              <Text>Repository: {selectedRepo?.owner?.login || selectedRepo?.owner}/{selectedRepo?.name}</Text>
            </Space>
            
            {/* Author Selector */}
            <Space style={{ width: '100%' }}>
              <TeamOutlined />
              <Text strong>Chọn tác giả commit:</Text>
              <Select
                style={{ minWidth: 300, flex: 1 }}
                placeholder="Chọn tác giả..."
                value={selectedAuthor}
                onChange={handleAuthorChange}
                loading={authorsLoading}
                showSearch
                optionFilterProp="children"
                filterOption={(input, option) =>
                  option?.label?.toLowerCase().indexOf(input.toLowerCase()) >= 0
                }
                options={availableAuthors.map(author => ({
                  value: author.author_name,
                  label: author.display_name,
                  title: `${author.author_name} - ${author.author_email} (${author.commit_count} commits)`
                }))}
              />
            </Space>
            
            {selectedAuthor && availableAuthors.length > 0 && (
              <Space style={{ marginTop: 8 }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {(() => {
                    const author = availableAuthors.find(a => a.author_name === selectedAuthor);
                    return author 
                      ? `📧 ${author.author_email} | 📝 ${author.commit_count} commits | 📅 Last: ${
                          author.last_commit_date 
                            ? new Date(author.last_commit_date).toLocaleDateString('vi-VN')
                            : 'N/A'
                        }`
                      : '';
                  })()}
                </Text>
              </Space>
            )}
          </Space>
        </Card>

        {/* Danh sách commit đã liên kết */}
        {linkedCommits.length > 0 && (
          <Card size="small" title="Commit đã liên kết" type="inner">
            <List
              size="small"
              dataSource={linkedCommits}
              renderItem={(commit) => (
                <List.Item>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>{commit.message}</Text>
                    <Space>
                      <Tag color="blue">{commit.sha?.substring(0, 8)}</Tag>
                      <Text type="secondary">
                        <UserOutlined /> {commit.author_name}
                      </Text>
                      <Text type="secondary">
                        <CalendarOutlined /> {formatDate(commit.committed_date)}
                      </Text>
                    </Space>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        )}

        {/* Tìm kiếm commit */}
        <Card size="small" title="Tìm kiếm và liên kết commit" type="inner">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Input.Search
              placeholder="Tìm kiếm commit theo message, SHA hoặc tác giả..."
              allowClear
              enterButton={<SearchOutlined />}
              loading={searchLoading}
              onSearch={handleSearch}
              style={{ marginBottom: 16 }}
            />

            {loading ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Spin size="large" />
                <div style={{ marginTop: '16px' }}>
                  <Text>Đang tải danh sách commit...</Text>
                </div>
              </div>
            ) : commits.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Text type="secondary">
                  {searchQuery 
                    ? `Không tìm thấy commit nào với từ khóa "${searchQuery}"`
                    : selectedAuthor
                      ? `Không tìm thấy commit nào của ${selectedAuthor}`
                      : 'Vui lòng chọn tác giả để xem commits'
                  }
                </Text>
              </div>
            ) : (
              <>
                <List
                  size="small"
                  dataSource={commits.filter(commit => !isCommitLinked(commit.sha))}
                  renderItem={(commit) => (
                    <List.Item>
                      <Checkbox
                        checked={selectedCommitIds.includes(commit.sha)}
                        onChange={(e) => handleCommitSelection(commit.sha, e.target.checked)}
                        disabled={isCommitLinked(commit.sha)}
                      >
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Text strong>{commit.message}</Text>
                          <Space>
                            <Tag color="green">{commit.sha?.substring(0, 8)}</Tag>
                            <Text type="secondary">
                              <UserOutlined /> {commit.author_name}
                            </Text>
                            <Text type="secondary">
                              <CalendarOutlined /> {formatDate(commit.committed_date)}
                            </Text>
                            <Text type="secondary">
                              +{commit.insertions} -{commit.deletions} files: {commit.files_changed}
                            </Text>
                          </Space>
                        </Space>
                      </Checkbox>
                    </List.Item>
                  )}
                />
                
                {/* Pagination */}
                {totalCommits > pageSize && (
                  <div style={{ textAlign: 'center', marginTop: 16 }}>
                    <Pagination
                      current={currentPage}
                      total={totalCommits}
                      pageSize={pageSize}
                      onChange={handlePageChange}
                      showSizeChanger={false}
                      showQuickJumper
                      showTotal={(total, range) => 
                        `${range[0]}-${range[1]} của ${total} commit`
                      }
                    />
                  </div>
                )}
                
                {/* Nút liên kết */}
                {selectedCommitIds.length > 0 && (
                  <div style={{ marginTop: '16px', textAlign: 'right' }}>
                    <Space>
                      <Text>Đã chọn: {selectedCommitIds.length} commit</Text>
                      <Button
                        type="primary"
                        icon={<LinkOutlined />}
                        loading={linking}
                        onClick={handleLinkCommits}
                      >
                        Liên kết Commit
                      </Button>
                    </Space>
                  </div>
                )}
              </>
            )}
          </Space>
        </Card>
      </Space>
    </Modal>
  );
};

export default TaskCommitLinker;
