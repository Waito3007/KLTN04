// src/pages/AuthSuccess.jsx
import React, { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";

const AuthSuccess = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
  const params = new URLSearchParams(location.search);
  const token = params.get("token");
  const email = params.get("email");
  const username = params.get("username");

  if (token) {
  const profile = {
    token,
    username,
    email,
    avatar_url: params.get("avatar_url"),
  };

  localStorage.setItem("github_profile", JSON.stringify(profile));
  localStorage.setItem("access_token", token); // ✅ THÊM DÒNG NÀY

  navigate("/dashboard");
} else {
  navigate("/login");
}

}, [location, navigate]);

  return (
    <div className="h-screen flex items-center justify-center">
      <p className="text-xl">Đang xác thực với GitHub...</p>
    </div>
  );
};

export default AuthSuccess;
