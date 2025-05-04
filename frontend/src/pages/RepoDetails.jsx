import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { message, Spin } from "antd";
import BranchSelector from "../components/Branchs/BranchSelector";
import CommitList from "../components/commits/CommitList";
import axios from "axios";

const RepoDetails = () => {
  const { owner, repo } = useParams();
  const [branch, setBranch] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const syncAllData = async () => {
      const token = localStorage.getItem("access_token");
      if (!token) {
        message.error("Vui lòng đăng nhập lại!");
        return;
      }

      try {
        setLoading(true);
        await axios.post(
          `http://localhost:8000/api/github/${owner}/${repo}/sync-all`,
          {},
          {
            headers: {
              Authorization: `token ${token}`,
            },
          }
        );
        message.success("Đồng bộ dữ liệu thành công!");
      } catch (error) {
        console.error("Lỗi khi đồng bộ dữ liệu:", error);
        message.error("Không thể đồng bộ dữ liệu!");
      } finally {
        setLoading(false);
      }
    };

    syncAllData();
  }, [owner, repo]);

  if (loading) {
    return <Spin tip="Đang đồng bộ dữ liệu..." size="large" />;
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ fontWeight: "bold" }}>📁 Repository: {repo}</h2>
      <BranchSelector owner={owner} repo={repo} onBranchChange={setBranch} />
      <CommitList owner={owner} repo={repo} branch={branch} />
    </div>
  );
};

export default RepoDetails;