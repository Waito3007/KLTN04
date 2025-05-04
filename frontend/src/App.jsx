import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import AuthSuccess from "./pages/AuthSuccess";
import Dashboard from "./pages/Dashboard"; // nếu có
import RepoDetails from "./pages/RepoDetails";
import CommitTable from './components/commits/CommitTable';

function App() {
  return (
    <Router>
      <Routes>
        {/* ✅ Trang mặc định là Login */}
        <Route path="/" element={<Navigate to="/login" />} />

        {/* Các route chính */}
        <Route path="/login" element={<Login />} />
        <Route path="/auth-success" element={<AuthSuccess />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/repo/:owner/:repo" element={<RepoDetails />} />
        <Route path="/commits" element={<CommitTable />} />

      </Routes>
    </Router>
  );
}

export default App;
