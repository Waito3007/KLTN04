//frontend\src\components\commits\CommitTable.jsxCommitTable.jsx

import { useEffect, useState } from 'react';
import { Table } from 'antd';
import axios from 'axios';

const CommitTable = () => {
  const [commits, setCommits] = useState([]);

  useEffect(() => {
    const fetchCommits = async () => {
      try {
        const response = await axios.get('http://localhost:8000/commits');
        setCommits(response.data);
      } catch (error) {
        console.error('Failed to fetch commits:', error);
      }
    };
    fetchCommits();
  }, []);

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
    },
    {
      title: 'Repo ID',
      dataIndex: 'repo_id',
    },
    {
      title: 'User ID',
      dataIndex: 'user_id',
    },
    {
      title: 'Message',
      dataIndex: 'message',
    },
    {
      title: 'Hash',
      dataIndex: 'commit_hash',
    },
    {
      title: 'Date',
      dataIndex: 'commit_date',
    },
  ];

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Lịch sử Commit</h2>
      <Table columns={columns} dataSource={commits} rowKey="id" />
    </div>
  );
};

export default CommitTable;