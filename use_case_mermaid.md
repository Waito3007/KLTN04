```mermaid
graph TB
    %% Actors
    DEV        %% Advanced Features
        UC24[🏷️ Filter Repository]
        UC25[� Responsive UI]
        UC26[🔔 Notification System]

        %% AI Models (Core Engine)
        UC27[🧠 HAN Model (Hierarchical Attention)]
        UC28[� CodeBERT Embeddings]
        UC29[🎯 Multi-task Learning]
    end💻 Developer]
    TL[👨‍💼 Team Leader]
    PM[👔 Project Manager]

    %% System boundary
    subgraph System["🎯 TaskFlowAI - Hệ thống Phân tích Commit & Quản lý Task"]
        %% Authentication & Core
        UC01[� GitHub OAuth Login]
        UC02[� Quản lý Profile User]
        UC03[📊 Dashboard Tổng quan]

        %% Repository Management
        UC04[� Xem Danh sách Repository]
        UC05[🔗 Kết nối Repository]
        UC06[� Chi tiết Repository]
        UC07[🌿 Quản lý Branch]

        %% Commit Analysis (AI Core)
        UC08[🤖 Phân tích Commit AI]
        UC09[� Phân loại Commit Message]
        UC10[📊 Thống kê Commit]
        UC11[� Tìm kiếm Commit]

        %% Task Management
        UC12[� Kanban Task Board]
        UC13[✅ Quản lý Assignment]
        UC14[🎯 Tạo/Chỉnh sửa Task]
        UC15[� Phân công Task]

        %% AI Insights & Suggestions
        UC16[� AI Insight Widget]
        UC17[🤖 Gợi ý Phân công Thông minh]
        UC18[⚠️ Cảnh báo Workload]
        UC19[� Dự đoán Tiến độ]

        %% Reporting & Analytics
        UC20[📊 Báo cáo Commit Analysis]
        UC21[� Metrics Overview Card]
        UC22[🔄 Export/Import Data]
        UC23[📋 Theo dõi Issues/PRs]

        %% Advanced Features
        UC24[🏷️ Filter Repository]
        UC25[� Responsive UI]
        UC26[🔔 Notification System]
    end

    %% External Systems
    GitHub[🐙 GitHub API]
    Database[🗄️ PostgreSQL Database]

    %% Developer connections (Basic User)
    DEV --> UC01
    DEV --> UC02
    DEV --> UC03
    DEV --> UC04
    DEV --> UC06
    DEV --> UC08
    DEV --> UC10
    DEV --> UC11
    DEV --> UC14

    %% Team Leader connections (Full Access)
    TL --> UC01
    TL --> UC02
    TL --> UC03
    TL --> UC04
    TL --> UC05
    TL --> UC06
    TL --> UC07
    TL --> UC08
    TL --> UC09
    TL --> UC10
    TL --> UC11
    TL --> UC12
    TL --> UC13
    TL --> UC14
    TL --> UC15
    TL --> UC16
    TL --> UC17
    TL --> UC18
    TL --> UC19
    TL --> UC20
    TL --> UC21
    TL --> UC22
    TL --> UC23
    TL --> UC24
    TL --> UC25
    TL --> UC26
    TL --> UC27
    TL --> UC28
    TL --> UC29

    %% Project Manager connections (Same as Team Leader)
    PM --> UC01
    PM --> UC02
    PM --> UC03
    PM --> UC04
    PM --> UC05
    PM --> UC06
    PM --> UC07
    PM --> UC08
    PM --> UC09
    PM --> UC10
    PM --> UC11
    PM --> UC12
    PM --> UC13
    PM --> UC14
    PM --> UC15
    PM --> UC16
    PM --> UC17
    PM --> UC18
    PM --> UC19
    PM --> UC20
    PM --> UC21
    PM --> UC22
    PM --> UC23
    PM --> UC24
    PM --> UC25
    PM --> UC26
    PM --> UC27
    PM --> UC28
    PM --> UC29

    %% System connections to external
    UC01 --> GitHub
    UC05 --> GitHub
    UC08 --> GitHub
    UC23 --> GitHub

    %% Database connections
    UC02 --> Database
    UC03 --> Database
    UC12 --> Database
    UC13 --> Database
    UC14 --> Database
    UC20 --> Database
    UC21 --> Database

    %% Include relationships
    UC03 -.->|includes| UC21
    UC03 -.->|includes| UC16
    UC12 -.->|includes| UC13
    UC15 -.->|includes| UC17
    UC08 -.->|includes| UC09
    UC08 -.->|includes| UC27
    UC08 -.->|includes| UC28
    UC17 -.->|includes| UC29
    UC20 -.->|includes| UC08
    UC16 -.->|includes| UC18

    %% Extend relationships
    UC17 -.->|extends| UC15
    UC18 -.->|extends| UC13
    UC19 -.->|extends| UC20
    UC24 -.->|extends| UC04
    UC27 -.->|extends| UC08
    UC28 -.->|extends| UC09

    classDef actor fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef core fill:#f3e5f5,stroke:#4a148c,stroke-width:1px
    classDef ai fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef management fill:#fff3e0,stroke:#e65100,stroke-width:1px
    classDef external fill:#ffebee,stroke:#c62828,stroke-width:1px
    classDef system fill:#f5f5f5,stroke:#424242,stroke-width:2px

    class DEV,TL,PM actor
    class UC01,UC02,UC03,UC04,UC05,UC06,UC07 core
    class UC08,UC09,UC16,UC17,UC18,UC19,UC27,UC28,UC29 ai
    class UC10,UC11,UC12,UC13,UC14,UC15,UC20,UC21,UC22,UC23,UC24,UC25,UC26 management
    class GitHub,Database external
    class System system
```
