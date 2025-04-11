import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import AuthSuccess from "./pages/AuthSuccess";
import Dashboard from "./pages/Dashboard"; // nếu có

function App() {
  return (
    <Router>
      <Routes>
        {/* ✅ Trang mặc định là Login */}
        <Route path="/" element={<Navigate to="/login" />} />

        {/* Các route chính */}
        <Route path="/login" element={<Login />} />
        <Route path="/api/auth-success" element={<AuthSuccess />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
