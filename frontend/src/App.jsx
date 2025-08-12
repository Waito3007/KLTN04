import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "@pages/Login";
import AuthSuccess from "@pages/AuthSuccess";
import DashboardModern from "@pages/DashboardModern";
import ComponentDemo from "@pages/ComponentDemo";
import RepoDetails from "@pages/RepoDetails";
import SyncPage from "@pages/SyncPage";
import RepositoryList from "@pages/RepositoryList";
import RepositoryAnalysis from "@pages/RepositoryAnalysis";
import CommitTable from "@components/commits/CommitTable";
import TestPage from "@pages/TestPage";
import ErrorBoundary from "@components/ErrorBoundary";
import RepoSyncManager from "@components/repo/RepoSyncManager";
import AppLayout from "@components/layout/AppLayout";

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <AppLayout>
          <Routes>
            {/* ✅ Trang mặc định là Login */}
            <Route path="/" element={<Login />} />

            {/* Các route chính */}
            <Route path="/login" element={<Login />} />
            <Route path="/auth-success" element={<AuthSuccess />} />
            <Route path="/dashboard" element={<DashboardModern />} />
            <Route path="/repositories" element={<RepositoryList />} />
            <Route path="/analysis" element={<RepositoryAnalysis />} />
            <Route path="/demo" element={<ComponentDemo />} />
            <Route path="/sync" element={<SyncPage />} />
            <Route path="/repo-sync" element={<RepoSyncManager />} />
            <Route path="/repo/:owner/:repo" element={<RepoDetails />} />
            <Route path="/repo-details" element={<RepoDetails />} />
            <Route path="/commits" element={<CommitTable />} />
            <Route path="/test" element={<TestPage />} />

            {/* Fallback route */}
            <Route path="*" element={<Navigate to="/login" replace />} />

          </Routes>
        </AppLayout>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
