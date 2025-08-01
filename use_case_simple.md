```mermaid
flowchart TD
    %% Actors
    DEV[    EXT1[🐙 GitHub API<br/>Repositories, Commits, PRs]
    EXT3[�️ PostgreSQL<br/>Data persistence & analytics]💻 Developer<br/>Lập trình viên]
    TL[👨‍💼 Team Leader<br/>Trưởng nhóm]
    PM[👔 Project Manager<br/>Quản lý dự án]

    %% Main System
    subgraph SYSTEM[🎯 TASKFLOWAI - HỆ THỐNG PHÂN TÍCH COMMIT & QUẢN LÝ TASK]
        direction TB

        %% Core Authentication & Dashboard
        subgraph CORE[Chức năng Cốt lõi]
            CORE1[� GitHub OAuth Login<br/>- Xác thực GitHub<br/>- Quản lý session<br/>- Profile user]
            CORE2[📊 Dashboard<br/>- Overview metrics<br/>- Repository list<br/>- Task summary]
            CORE3[📂 Repository Management<br/>- Kết nối GitHub/GitLab<br/>- Browse repositories<br/>- Repository details]
        end

        %% AI Analysis Engine
        subgraph AI_ENGINE[AI Engine - Phân tích Thông minh]
            AI1[🤖 Commit Analysis AI<br/>- XGBoost Classification<br/>- Message categorization<br/>- Code change analysis]
            AI2[💡 Task Assignment AI<br/>- Workload balancing<br/>- Skill matching<br/>- Smart suggestions]
            AI3[� Prediction & Insights<br/>- Progress forecasting<br/>- Risk analysis<br/>- Performance trends]
            AI4[⚠️ Alert System<br/>- Workload warnings<br/>- Deadline alerts<br/>- Quality concerns]
        end

        %% Developer Functions
        subgraph DEV_FUNC[Chức năng Developer]
            DEV1[📋 Personal Task View<br/>- Assigned tasks<br/>- Personal progress<br/>- Commit history]
            DEV2[📊 Commit Analytics<br/>- Personal statistics<br/>- Code quality metrics<br/>- Contribution graphs]
        end

        %% Team Leader Functions
        subgraph TL_FUNC[Chức năng Team Leader]
            TL1[� Team Management<br/>- Member overview<br/>- Workload distribution<br/>- Performance tracking]
            TL2[📋 Kanban Task Board<br/>- Drag-drop interface<br/>- Task assignment<br/>- Status tracking]
            TL3[🎯 Assignment Control<br/>- Manual assignment<br/>- AI-assisted allocation<br/>- Workload balancing]
            TL4[📈 Team Analytics<br/>- Team velocity<br/>- Bottleneck detection<br/>- Progress monitoring]
        end

        %% Project Manager Functions
        subgraph PM_FUNC[Chức năng Project Manager]
            PM1[� Executive Dashboard<br/>- Multi-project view<br/>- High-level metrics<br/>- Strategic insights]
            PM2[📈 Advanced Analytics<br/>- Cross-team comparison<br/>- Resource utilization<br/>- ROI analysis]
            PM3[🔮 Strategic Planning<br/>- Timeline prediction<br/>- Resource allocation<br/>- Risk management]
            PM4[� Reporting System<br/>- Export capabilities<br/>- Custom reports<br/>- Stakeholder updates]
        end
    end

    %% External Systems
    EXT1[🐙 GitHub API<br/>Repositories, Commits, PRs]
    EXT2[🦊 GitLab API<br/>Alternative Git platform]
    EXT3[�️ PostgreSQL<br/>Data persistence]

    %% Actor Connections
    DEV --> CORE1
    DEV --> CORE2
    DEV --> CORE3
    DEV --> DEV1
    DEV --> DEV2
    DEV --> AI1

    TL --> CORE1
    TL --> CORE2
    TL --> CORE3
    TL --> TL1
    TL --> TL2
    TL --> TL3
    TL --> TL4
    TL --> AI1
    TL --> AI2
    TL --> AI3
    TL --> AI4

    PM --> CORE1
    PM --> CORE2
    PM --> CORE3
    PM --> PM1
    PM --> PM2
    PM --> PM3
    PM --> PM4
    PM --> AI1
    PM --> AI2
    PM --> AI3
    PM --> AI4

    %% External Connections
    CORE1 --> EXT1
    CORE3 --> EXT1
    AI1 --> EXT1
    SYSTEM --> EXT3

    %% Internal Dependencies
    TL2 -.-> AI2
    TL3 -.-> AI2
    TL4 -.-> AI1
    PM2 -.-> AI1
    PM3 -.-> AI3
    AI4 -.-> TL1
    AI4 -.-> PM1

    %% Styling
    style DEV fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style TL fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style PM fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style SYSTEM fill:#f5f5f5,stroke:#424242,stroke-width:3px

    style CORE fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style AI_ENGINE fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style DEV_FUNC fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style TL_FUNC fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style PM_FUNC fill:#fff8e1,stroke:#ff8f00,stroke-width:2px

    style EXT1 fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style EXT3 fill:#e0f2f1,stroke:#00695c,stroke-width:2px
```
