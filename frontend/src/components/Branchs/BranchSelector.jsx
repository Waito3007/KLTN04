import { useEffect, useState } from "react";
import { Select, Spin, message, Tag, Typography, Divider } from "antd";
import { GithubOutlined, BranchesOutlined } from '@ant-design/icons';
import axios from "axios";
import styled from "styled-components";

const { Option } = Select;
const { Text } = Typography;

const SelectContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const StyledSelect = styled(Select)`
  min-width: 240px;
  
  .ant-select-selector {
    border-radius: 6px !important;
    border: 1px solid #d9d9d9 !important;
    transition: all 0.3s !important;
    
    &:hover {
      border-color: #1890ff !important;
    }
  }
  
  .ant-select-selection-item {
    font-weight: 500;
  }
`;

const BranchTag = styled(Tag)`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 6px;
  background: #f0f5ff;
  color: #1890ff;
  border: 1px solid #d6e4ff;
`;

const BranchSelector = ({ owner, repo, onBranchChange }) => {
  const [branches, setBranches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedBranch, setSelectedBranch] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) return;

    const fetchBranches = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8000/api/github/${owner}/${repo}/branches`,
          {
            headers: {
              Authorization: `token ${token}`,
            },
          }
        );
        setBranches(response.data);
        if (response.data.length > 0) {
          setSelectedBranch(response.data[0].name);
          onBranchChange(response.data[0].name);
        }
      } catch (err) {
        console.error(err);
        message.error("Không lấy được danh sách branch");
      } finally {
        setLoading(false);
      }
    };

    fetchBranches();
  }, [owner, repo]);

  const handleChange = (value) => {
    setSelectedBranch(value);
    onBranchChange(value);
  };

  if (loading) return <Spin size="small" />;

  return (
    <div style={{ marginBottom: 16 }}>
      {/* <Divider orientation="left" style={{ fontSize: 32, color: '#666' }}>
        Chọn branch
      </Divider> */}
      
      <SelectContainer>
        <BranchTag>
          <BranchesOutlined />
          <Text strong>Branch:</Text>
        </BranchTag>
        
        <StyledSelect
          value={selectedBranch}
          onChange={handleChange}
          suffixIcon={<GithubOutlined style={{ color: '#1890ff' }} />}
          dropdownMatchSelectWidth={false}
        >
          {branches.map((branch) => (
            <Option key={branch.name} value={branch.name}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <BranchesOutlined style={{ color: '#52c41a' }} />
                <Text strong>{branch.name}</Text>
              </div>
            </Option>
          ))}
        </StyledSelect>
      </SelectContainer>
    </div>
  );
};

export default BranchSelector;