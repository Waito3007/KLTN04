# TaskFlowAI - Single Actor Use Case Model

## 🎯 Simplified Actor Model

Hệ thống TaskFlowAI được thiết kế với **single actor approach** - tất cả users được coi là **Team Leader/Project Manager** với quyền truy cập đầy đủ.

## 👤 Actor Definition

### Team Leader/Project Manager

**Mô tả**: Người quản lý dự án và lãnh đạo nhóm phát triển - không phân biệt vai trò cụ thể
**Quyền truy cập**: Full access đến tất cả 29 use cases

## 📋 Use Case Summary

### Core Use Cases (29 total)

| ID          | Use Case          | Mô tả ngắn                                       |
| ----------- | ----------------- | ------------------------------------------------ |
| **UC01-07** | Core & Auth       | Login, profile, dashboard, repository management |
| **UC08-11** | Commit Analysis   | AI-powered commit analysis với HAN + CodeBERT    |
| **UC12-15** | Task Management   | Kanban board, assignments, task creation         |
| **UC16-19** | AI Insights       | Smart suggestions, workload alerts, predictions  |
| **UC20-23** | Reporting         | Analytics, export, tracking issues/PRs           |
| **UC24-26** | Advanced Features | Filters, responsive UI, notifications            |
| **UC27-29** | AI Models         | HAN, CodeBERT, multi-task learning engine        |

## 🔄 Simplified User Journey

```
GitHub OAuth → Dashboard → Repository Selection → AI Analysis → Task Management → Reporting
```

**All users can:**

- ✅ Access all features without restrictions
- ✅ Manage repositories and teams
- ✅ Use AI analysis and insights
- ✅ Create and assign tasks
- ✅ Generate reports and analytics
- ✅ Configure system settings

## 🎨 UI/UX Benefits

### Single Interface Design

- **One dashboard** for all users
- **No role switching** or permission complexity
- **Consistent navigation** and features
- **Simplified user experience**

### Development Benefits

- **No RBAC complexity** - single permission model
- **Easier testing** - one user flow to validate
- **Simpler codebase** - no role-based branching
- **Faster development** - no permission edge cases

## 🔧 Technical Implementation

### Backend Simplification

```python
# No role-based access control needed
@app.get("/api/*")
async def any_endpoint(user: User = Depends(get_current_user)):
    # Same logic for all authenticated users
    return data

# Simplified user model
class User:
    id: int
    github_username: str
    email: str
    # No role field needed
```

### Frontend Simplification

```jsx
// Single component set for all users
function App() {
  return (
    <Dashboard />      // Available to all
    <TaskBoard />      // Available to all
    <AIInsights />     // Available to all
    <Reports />        // Available to all
  )
}
```

## 📊 Actor-Use Case Matrix

| Use Case Categories             | Team Leader/Project Manager |
| ------------------------------- | :-------------------------: |
| Authentication & Core (UC01-07) |       ✅ Full Access        |
| AI Commit Analysis (UC08-11)    |       ✅ Full Access        |
| Task Management (UC12-15)       |       ✅ Full Access        |
| AI Insights (UC16-19)           |       ✅ Full Access        |
| Reporting (UC20-23)             |       ✅ Full Access        |
| Advanced Features (UC24-26)     |       ✅ Full Access        |
| AI Models (UC27-29)             |       ✅ Full Access        |

## 🎯 Focus Areas

With simplified actor model, development focuses on:

1. **🚀 Feature Quality**: Better implementation of core features
2. **🎨 User Experience**: Intuitive, consistent interface
3. **🧠 AI Performance**: Optimized HAN + CodeBERT models
4. **📈 System Performance**: Fast, responsive application
5. **🔧 Reliability**: Robust GitHub integration and data handling

## 📝 Documentation Structure

- `use_case_diagram.md` - Main use case documentation
- `use_case_mermaid_single_actor.md` - Mermaid diagram with single actor
- `unified_roles_analysis.md` - Analysis of unified approach
- `ai_models_analysis.md` - AI models technical details

---

_TaskFlowAI: Simplified Single Actor Model for Universal Access_
