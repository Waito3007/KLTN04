import { useEffect, useState } from "react";

const CommitList = ({ owner, repo, branch }) => {
  const [commits, setCommits] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCommits = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/github/${owner}/${repo}/commits`);
        const data = await response.json();
        setCommits(data);
      } catch (error) {
        console.error("Error fetching commits:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchCommits();
  }, [owner, repo, branch]);

  if (loading) return <p>Loading...</p>;

  return (
    <ul>
      {commits.map((commit) => (
        <li key={commit.sha}>
          <p>{commit.message}</p>
          <p>{commit.author_name}</p>
          <p>{commit.date}</p>
        </li>
      ))}
    </ul>
  );
};

export default CommitList;
