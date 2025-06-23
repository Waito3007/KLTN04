import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { message, Button, Card, Typography, Alert, Progress, Row, Col } from "antd";
import { SyncOutlined, SaveOutlined, GithubOutlined } from "@ant-design/icons";
import BranchSelector from "../components/Branchs/BranchSelector";
import BranchCommitList from "../components/Branchs/BranchCommitList";
import CommitList from "../components/commits/CommitList";
import axios from "axios";

const { Title, Text } = Typography;

const RepoDetails = () => {
  const { owner, repo } = useParams();
  const [branch, setBranch] = useState("");
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncProgress, setSyncProgress] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0); // For refreshing child components

  // Handle branch change with optional refresh
  const handleBranchChange = (newBranch, shouldRefresh = false) => {
    setBranch(newBranch);
    if (shouldRefresh) {
      setRefreshKey(prev => prev + 1); // Trigger refresh
    }
  };

  // Sync repository trong background không block UI
  const syncRepositoryInBackground = useCallback(async () => {
    const token = localStorage.getItem("access_token");
    if (!token || syncing) return;

    try {
      setSyncing(true);
      setSyncProgress(0);
      
      // Hiển thị thông báo bắt đầu sync
      message.info(`Đang đồng bộ repository ${repo} trong background...`, 2);
      
      // Sync cơ bản trước (nhanh)
      setSyncProgress(30);
      await axios.post(
        `http://localhost:8000/api/github/${owner}/${repo}/sync-basic`,
        {},
        {
          headers: { Authorization: `token ${token}` },
        }
      );
      
      // Sync đầy đủ
      setSyncProgress(70);
      await axios.post(
        `http://localhost:8000/api/github/${owner}/${repo}/sync-all`,
        {},
        {
          headers: { Authorization: `token ${token}` },
        }
      );
      
      setSyncProgress(100);
      message.success(`Đồng bộ repository ${repo} thành công!`);
      
    } catch (error) {
      console.error("Lỗi khi đồng bộ repository:", error);
      message.error("Đồng bộ repository thất bại!");
    } finally {
      setSyncing(false);
      setTimeout(() => setSyncProgress(0), 2000);
    }
  }, [owner, repo, syncing]);

  // Kiểm tra và sync repository trong background
  const checkAndSyncRepository = useCallback(async () => {
    const token = localStorage.getItem("access_token");
    if (!token) return;

    try {
      // Kiểm tra xem repo đã có dữ liệu chưa
      const checkResponse = await axios.get(
        `http://localhost:8000/api/github/${owner}/${repo}/branches`,
        {
          headers: { Authorization: `token ${token}` },
        }
      );

      // Nếu có dữ liệu rồi thì không cần sync
      if (checkResponse.data && checkResponse.data.length > 0) {
        console.log('Repository đã có dữ liệu, không cần sync');
        return;
      }
    } catch {
      console.log('Repository chưa có dữ liệu, bắt đầu sync...');
    }

    // Sync repository trong background
    syncRepositoryInBackground();
  }, [owner, repo, syncRepositoryInBackground]);

  // Load trang ngay lập tức với dữ liệu có sẵn
  useEffect(() => {
    // Sync trong background nếu cần
    checkAndSyncRepository();
  }, [owner, repo, checkAndSyncRepository]);

  // Sync thủ công
  const manualSync = async () => {
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
          headers: { Authorization: `token ${token}` },
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
  // const saveCommits = async () => {
  //   const token = localStorage.getItem("access_token");
  //   if (!token) {
  //     message.error("Vui lòng đăng nhập lại!");
  //     return;
  //   }

  //   if (!branch) {
  //     message.error("Vui lòng chọn branch trước!");
  //     return;
  //   }

  //   try {
  //     const response = await axios.post(
  //       `http://localhost:8000/api/github/${owner}/${repo}/branches/${branch}/sync-commits?include_stats=true&per_page=100&max_pages=5`,
  //       {},
  //       {
  //         headers: {
  //           Authorization: `token ${token}`,
  //         },
  //       }
  //     );
      
  //     const { stats } = response.data;
  //     message.success(
  //       `Đồng bộ thành công! ${stats.new_commits_saved} commits mới được lưu cho branch "${branch}"`
  //     );
  //   } catch (error) {
  //     console.error("Lỗi khi lưu commit:", error);
  //     message.error("Không thể lưu commit!");
  //   }
  // };

  // Hiển thị trang ngay lập tức, không đợi sync
  return (
    <div style={{ padding: 24 }}>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Title level={2} style={{ margin: 0 }}>
            <GithubOutlined /> {owner}/{repo}
          </Title>
          
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {syncing && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Progress 
                  type="circle" 
                  size={24} 
                  percent={syncProgress}
                  showInfo={false}
                />
                <Text type="secondary">Đang đồng bộ...</Text>
              </div>
            )}
            
            <Button 
              icon={<SyncOutlined />} 
              onClick={manualSync}
              loading={loading}
              disabled={loading || syncing}
            >
              Đồng bộ thủ công
            </Button>
            
            {/* <Button 
              type="primary" 
              icon={<SaveOutlined />} 
              onClick={saveCommits}
              disabled={!branch}
            >
              Lưu Commit
            </Button> */}
          </div>
        </div>

        {syncing && (
          <Alert
            message="Đang đồng bộ dữ liệu trong background"
            description="Bạn có thể tiếp tục sử dụng trang này, việc đồng bộ sẽ hoàn thành trong giây lát."
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />        )}        <BranchSelector owner={owner} repo={repo} onBranchChange={handleBranchChange} />
      </Card>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <BranchCommitList 
            key={`branch-commits-${refreshKey}`}
            owner={owner} 
            repo={repo} 
            selectedBranch={branch} 
          />
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Commits từ GitHub API (Real-time)">
            <CommitList 
              key={`real-time-commits-${refreshKey}`}
              owner={owner} 
              repo={repo} 
              branch={branch} 
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default RepoDetails;
