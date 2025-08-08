# Frontend Setup Guide

This guide provides instructions on how to set up and run the frontend application. The frontend is built with React and Vite.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Node.js (LTS version recommended)**: [Download Node.js](https://nodejs.org/en/download/)
*   **npm** (comes with Node.js) or **Yarn** (optional): A package manager for JavaScript.
*   **Git**: For cloning the repository. [Git Downloads](https://git-scm.com/downloads)

## 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone <repository_url>
cd KLTN04/frontend
```

## 2. Install Dependencies

Navigate to the `frontend` directory and install the necessary Node.js packages:

```bash
cd E:/Dự Án Của Nghĩa/KLTN04/frontend
npm install
# Or if you use Yarn:
# yarn install
```

## 3. Configure Backend API Endpoint

The frontend needs to know where your backend API is running. You might need to configure this in `frontend/src/config.js` or similar configuration file. Look for a variable like `VITE_API_BASE_URL` or `REACT_APP_API_URL`.

If your backend is running on `http://localhost:8000`, ensure your frontend configuration points to this URL.

Example `frontend/src/config.js` (adjust as needed):

```javascript
// frontend/src/config.js
const config = {
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
};

export default config;
```

If you are using environment variables, create a `.env` file in the `frontend` directory (`E:/Dự Án Của Nghĩa/KLTN04/frontend/.env`):

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

## 4. Run the Frontend Application

Start the development server:

```bash
npm run dev
# Or if you use Yarn:
# yarn dev
```

This will typically start the frontend application on `http://localhost:5173` (or another available port). The console output will show you the exact URL.

## 5. Build for Production (Optional)

To create a production-ready build of the frontend application:

```bash
npm run build
# Or if you use Yarn:
# yarn build
```

This will generate optimized static assets in the `dist` directory, which can then be served by a web server (e.g., Nginx, Apache) or integrated with the backend.
