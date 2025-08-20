import React, { Suspense } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { Spin } from "antd";
import ErrorBoundary from "@components/ErrorBoundary";
import AppLayout from "@components/layout/AppLayout";
import { AuthProvider } from "@components/layout/AuthContext";
import ProtectedRoute from "@components/auth/ProtectedRoute";
import PublicRoute from "@components/auth/PublicRoute";

// Sử dụng React.lazy để import động các component
const Login = React.lazy(() => import("@pages/Login"));
const AuthSuccess = React.lazy(() => import("@pages/AuthSuccess"));
const DashboardModern = React.lazy(() => import("@pages/DashboardModern"));
const ComponentDemo = React.lazy(() => import("@pages/ComponentDemo"));
const RepoDetails = React.lazy(() => import("@pages/RepoDetails"));
const SyncPage = React.lazy(() => import("@pages/SyncPage"));
const RepositoryList = React.lazy(() => import("@pages/RepositoryList"));
const RepositoryAnalysis = React.lazy(() => import("@pages/RepositoryAnalysis"));
const CommitTable = React.lazy(() => import("@components/commits/CommitTable"));
const TestPage = React.lazy(() => import("@pages/TestPage"));
const RepoSyncManager = React.lazy(() => import("@components/repo/RepoSyncManager"));
const SyncManagerPage = React.lazy(() => import("@pages/SyncManagerPage"));

// Component Loading được cải thiện
const LoadingSpinner = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '50vh',
    position: 'relative',
  }}>
    <div style={{ position: 'absolute', zIndex: 1 }}>
      <Spin size="large" tip="Đang tải..." />
    </div>
  </div>
);

function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <Router>
          <AppLayout>
            <Suspense fallback={<LoadingSpinner />}>
              <Routes>
                {/* ✅ Public Routes - chỉ truy cập được khi chưa đăng nhập */}
                <Route path="/" element={
                  <PublicRoute>
                    <Login />
                  </PublicRoute>
                } />
                
                <Route path="/login" element={
                  <PublicRoute>
                    <Login />
                  </PublicRoute>
                } />
                
                <Route path="/auth-success" element={
                  <PublicRoute redirectTo="/dashboard">
                    <AuthSuccess />
                  </PublicRoute>
                } />

                {/* ✅ Protected Routes - chỉ truy cập được khi đã đăng nhập */}
                <Route path="/dashboard" element={
                  <ProtectedRoute>
                    <DashboardModern />
                  </ProtectedRoute>
                } />
                
                <Route path="/repositories" element={
                  <ProtectedRoute>
                    <RepositoryList />
                  </ProtectedRoute>
                } />
                
                <Route path="/analysis" element={
                  <ProtectedRoute>
                    <RepositoryAnalysis />
                  </ProtectedRoute>
                } />
                
                <Route path="/demo" element={
                  <ProtectedRoute>
                    <ComponentDemo />
                  </ProtectedRoute>
                } />
                
                <Route path="/sync" element={
                  <ProtectedRoute>
                    <SyncPage />
                  </ProtectedRoute>
                } />
                
                <Route path="/repo-sync" element={
                  <ProtectedRoute>
                    <RepoSyncManager />
                  </ProtectedRoute>
                } />
                
                <Route path="/sync-manager" element={
                  <ProtectedRoute>
                    <SyncManagerPage />
                  </ProtectedRoute>
                } />
                
                <Route path="/repo/:owner/:repo" element={
                  <ProtectedRoute>
                    <RepoDetails />
                  </ProtectedRoute>
                } />
                
                <Route path="/repo-details" element={
                  <ProtectedRoute>
                    <RepoDetails />
                  </ProtectedRoute>
                } />
                
                <Route path="/commits" element={
                  <ProtectedRoute>
                    <CommitTable />
                  </ProtectedRoute>
                } />
                
                <Route path="/test" element={
                  <ProtectedRoute>
                    <TestPage />
                  </ProtectedRoute>
                } />

                {/* Fallback route */}
                <Route path="*" element={<Navigate to="/login" replace />} />
              </Routes>
            </Suspense>
          </AppLayout>
        </Router>
      </AuthProvider>
    </ErrorBoundary>
  );
}

export default App;
