// src/pages/AuthSuccess.jsx
import React, { useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Toast } from '@components/common';
import { useAuth } from '@components/layout/useAuth';
import MainLayout from '@components/layout/MainLayout';

const AuthSuccess = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();

  // Flag để đảm bảo Toast và login chỉ chạy 1 lần
  const didLogin = useRef(false);

  useEffect(() => {
    if (didLogin.current) return;
    didLogin.current = true;

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
      try {
        login(profile);
        localStorage.setItem("access_token", token);
        setTimeout(() => {
          Toast.success("Đăng nhập thành công!");
          navigate("/dashboard");
        }, 100);
      } catch (error) {
        console.error("Lỗi khi đăng nhập:", error);
        Toast.error("Có lỗi xảy ra khi đăng nhập!");
        navigate("/login");
      }
    } else {
      Toast.error("Không tìm thấy thông tin xác thực!");
      navigate("/login");
    }
  }, [location, navigate, login]);

  return (
    <MainLayout variant="glass" centered={true}>
      <div style={{
        textAlign: 'center',
        padding: '40px',
        background: 'rgba(255, 255, 255, 0.9)',
        borderRadius: '20px',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <div style={{
          fontSize: '48px',
          marginBottom: '20px',
          background: 'linear-gradient(45deg, #1890ff, #52c41a)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text'
        }}>
          ✨
        </div>
        <h2 style={{
          fontSize: '24px',
          fontWeight: 600,
          color: '#262626',
          marginBottom: '16px'
        }}>
          Đang đồng bộ dữ liệu...
        </h2>
        <p style={{
          fontSize: '16px',
          color: '#595959',
          margin: 0
        }}>
          Vui lòng chờ trong giây lát
        </p>
        <div style={{
          marginTop: '24px',
          width: '200px',
          height: '4px',
          background: '#f0f0f0',
          borderRadius: '2px',
          overflow: 'hidden',
          margin: '24px auto 0'
        }}>
          <div style={{
            width: '100%',
            height: '100%',
            background: 'linear-gradient(45deg, #1890ff, #52c41a)',
            animation: 'loading 2s infinite'
          }} />
        </div>
      </div>
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes loading {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
        `
      }} />
    </MainLayout>
  );
};

export default AuthSuccess;