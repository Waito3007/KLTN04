// src/pages/AuthSuccess.jsx
import React, { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { message } from "antd";
import axios from "axios";

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
      };

      localStorage.setItem("github_profile", JSON.stringify(profile));
      localStorage.setItem("access_token", token);

      const syncAllRepositories = async () => {
        try {
          const response = await axios.get("http://localhost:8000/api/github/repos", {
            headers: {
              Authorization: `token ${token}`,
            },
          });

          const repositories = response.data;
          for (const repo of repositories) {
            await axios.post(
              `http://localhost:8000/api/github/${repo.owner.login}/${repo.name}/sync-all`,
              {},
              {
                headers: {
                  Authorization: `token ${token}`,
                },
              }
            );
          }

          message.success("Đồng bộ dữ liệu thành công!");
        } catch (error) {
          console.error("Lỗi khi đồng bộ repository:", error);
          message.error("Không thể đồng bộ repository!");
        }
      };

      syncAllRepositories();
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