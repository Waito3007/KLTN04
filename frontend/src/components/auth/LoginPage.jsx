import { useState } from "react";
import { FaGithub } from "react-icons/fa";

function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-tr from-blue-100 to-indigo-200 px-4">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl p-8 space-y-6">
        <h2 className="text-3xl font-extrabold text-center text-gray-800">
          Chào mừng quay lại
        </h2>
        <p className="text-center text-gray-500 text-sm">
          Vui lòng đăng nhập để tiếp tục
        </p>

        <form className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email
            </label>
            <input
              type="email"
              value={email}
              placeholder="you@example.com"
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none transition"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Mật khẩu
            </label>
            <input
              type="password"
              value={password}
              placeholder="••••••••"
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none transition"
            />
          </div>

          <button
            type="submit"
            className="w-full bg-indigo-600 text-white py-2 rounded-lg font-medium hover:bg-indigo-700 transition"
          >
            Đăng nhập
          </button>
        </form>

        <div className="text-center">
          <p className="text-sm text-gray-500">hoặc</p>
        </div>

        <button className="w-full flex items-center justify-center gap-2 py-2 border border-gray-300 rounded-lg hover:bg-gray-100 transition text-gray-700 font-medium">
          <FaGithub size={18} />
          Đăng nhập với GitHub
        </button>

        <p className="text-center text-sm text-gray-500">
          Bạn chưa có tài khoản?{" "}
          <a
            href="#"
            className="text-indigo-600 hover:underline font-medium"
          >
            Đăng ký
          </a>
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
