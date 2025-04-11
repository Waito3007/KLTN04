import { useParams } from "react-router-dom";
import { useState } from "react";
import BranchSelector from "../components/Branchs/BranchSelector";
import CommitList from "../components/commits/CommitList";

const RepoDetails = () => {
  const { owner, repo } = useParams(); // nếu em dùng react-router
  const [branch, setBranch] = useState("");

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ fontWeight: "bold" }}>📁 Repository: {repo}</h2>
      <BranchSelector owner={owner} repo={repo} onBranchChange={setBranch} />
      <CommitList owner={owner} repo={repo} branch={branch} />
    </div>
  );
};

export default RepoDetails;
