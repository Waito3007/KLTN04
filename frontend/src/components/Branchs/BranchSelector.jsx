import { useEffect, useState } from "react";
import { Select, Spin, message } from "antd";
import axios from "axios";

const BranchSelector = ({ owner, repo, onBranchChange }) => {
  const [branches, setBranches] = useState([]);
  const [loading, setLoading] = useState(true);

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
        onBranchChange(response.data[0]?.name); // auto chọn branch đầu tiên
      } catch (err) {
        console.error(err);
        message.error("Không lấy được danh sách branch");
      } finally {
        setLoading(false);
      }
    };

    fetchBranches();
  }, [owner, repo]);

  if (loading) return <Spin size="small" />;

  return (
    <Select
      style={{ width: 200 }}
      onChange={onBranchChange}
      defaultValue={branches[0]?.name}
    >
      {branches.map((branch) => (
        <Select.Option key={branch.name} value={branch.name}>
          {branch.name}
        </Select.Option>
      ))}
    </Select>
  );
};

export default BranchSelector;
