// src/pages/AuthSuccess.jsx
import React, { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { message } from "antd";

const AuthSuccess = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get("token");
    const username = params.get("username");
    const email = params.get("email");

    if (token) {
      const profile = {
        token,
        username,
        email,
        avatar_url: params.get("avatar_url"),
      };      localStorage.setItem("github_profile", JSON.stringify(profile));
      localStorage.setItem("access_token", token);

      // Chuyển hướng ngay lập tức, để Dashboard xử lý đồng bộ
      message.success("Đăng nhập thành công!");
      navigate("/dashboard");
    } else {
      navigate("/login");
    }
  }, [location, navigate]);

  return (
    <div className="h-screen flex items-center justify-center">
      <p className="text-xl">Đang đồng bộ dữ liệu...</p>
    </div>
  );
};

export default AuthSuccess;