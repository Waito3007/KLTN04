# TaskFlowAI - Unified User Roles

## 🎯 Approach: Unified Access Model

Hệ thống TaskFlowAI được thiết kế với **access model thống nhất** cho cả Team Leader và Project Manager, không phân biệt quyền hạn hay chức năng.

## 👥 Actor Roles (Equivalent Access)

### Team Leader & Project Manager

**Cùng quyền truy cập và chức năng:**

| Nhóm chức năng               | Mô tả                                      | Use Cases                    |
| ---------------------------- | ------------------------------------------ | ---------------------------- |
| **🔐 Authentication**        | Đăng nhập và quản lý profile               | UC01, UC02                   |
| **📊 Dashboard & Analytics** | Dashboard tổng quan với AI insights        | UC03, UC21, UC16             |
| **📂 Repository Management** | Quản lý repositories và GitHub integration | UC04, UC05, UC06, UC07       |
| **🤖 AI-Powered Analysis**   | Commit analysis với HAN + CodeBERT         | UC08, UC09, UC27, UC28, UC29 |
| **📋 Task Management**       | Kanban board và assignment management      | UC12, UC13, UC14, UC15       |
| **💡 AI Suggestions**        | Smart assignment và workload optimization  | UC17, UC18, UC19             |
| **📈 Reporting & Export**    | Báo cáo và export dữ liệu                  | UC20, UC22, UC23             |
| **⚙️ System Features**       | Filters, notifications, responsive UI      | UC24, UC25, UC26             |

## 🔄 Unified User Journey

```
GitHub OAuth Login → Dashboard → Repository Selection → AI Analysis → Task Management → Reporting
```

**Cả Team Leader và Project Manager đều có thể:**

1. ✅ Xem và quản lý tất cả repositories
2. ✅ Truy cập đầy đủ AI analysis và insights
3. ✅ Phân công và quản lý tasks
4. ✅ Tạo và export báo cáo
5. ✅ Cấu hình alerts và notifications
6. ✅ Sử dụng tất cả AI features

## 🎨 UI/UX Implications

### Single Dashboard Design

- **Không có role-specific views**
- **Same navigation menu** cho cả hai actors
- **Identical feature access** across all components
- **Unified permission model** in backend

### Component Access

```jsx
// Không cần role checking
<Dashboard /> // Available to both TL & PM
<TaskBoard /> // Available to both TL & PM
<AIInsights /> // Available to both TL & PM
<Reports /> // Available to both TL & PM
```

## 🔧 Backend Implementation

### No Role-Based Access Control (RBAC)

```python
# Simplified auth - no role differentiation
@app.get("/api/dashboard")
async def get_dashboard(user: User = Depends(get_current_user)):
    # Same response for all authenticated users
    return dashboard_data

@app.get("/api/tasks")
async def get_tasks(user: User = Depends(get_current_user)):
    # No role-based filtering
    return all_tasks
```

### Database Schema

```sql
-- Users table without role distinction
users (
    id, github_id, github_username, email, avatar_url
    -- No 'role' or 'permissions' columns needed
)
```

## 📊 Use Case Statistics

**Total Use Cases**: 29

- **Shared by both actors**: 29 (100%)
- **Team Leader exclusive**: 0 (0%)
- **Project Manager exclusive**: 0 (0%)

## 💡 Benefits of Unified Approach

1. **Simplified Development**: Single codebase, no role branching
2. **Better UX**: Consistent experience regardless of title
3. **Easier Maintenance**: No complex permission logic
4. **Flexible Usage**: Users can wear multiple hats
5. **Faster Onboarding**: Same learning curve for everyone

## 🎯 Focus Areas

Since roles are unified, the system focuses on:

- **📈 Data Quality**: Better AI insights for all users
- **🚀 Performance**: Optimized for high usage
- **🎨 Usability**: Intuitive interface design
- **🔧 Reliability**: Robust GitHub integration
- **🧠 AI Accuracy**: High-quality HAN + CodeBERT models

---

_TaskFlowAI: One System, Universal Access for Team Leaders & Project Managers_
