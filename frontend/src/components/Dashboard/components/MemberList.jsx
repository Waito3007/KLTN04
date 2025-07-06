import React from 'react';
import { Card, Avatar, List, Typography, Empty } from 'antd';
import { UserOutlined } from '@ant-design/icons';

const { Text } = Typography;

/**
 * Component hiển thị danh sách thành viên repository
 */
const MemberList = ({ members, loading, selectedMember, onMemberClick }) => {
  return (
    <Card title="👥 Danh sách thành viên" loading={loading}>
      {members.length === 0 ? (
        <Empty description="Không có thành viên nào" />
      ) : (
        <List
          dataSource={members}
          renderItem={member => (
            <List.Item
              style={{
                cursor: 'pointer',
                padding: '12px',
                backgroundColor: selectedMember?.login === member.login ? '#e6f7ff' : 'transparent',
                borderRadius: '6px',
                marginBottom: '8px'
              }}
              onClick={() => onMemberClick(member)}
            >
              <List.Item.Meta
                avatar={
                  <Avatar 
                    src={member.avatar_url} 
                    icon={<UserOutlined />}
                    size="large"
                  />
                }
                title={member.display_name}
                description={
                  <div>
                    <div>@{member.login}</div>
                    <Text type="secondary">{member.total_commits} commits</Text>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      )}
    </Card>
  );
};

export default MemberList;
