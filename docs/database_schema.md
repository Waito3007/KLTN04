## Bảng `users`

| TT  | Tên thuộc tính     | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị     | Diễn giải                  |
| --- | ------------------ | ------------ | ------- | -------- | --------------------------- | -------------------------- |
| 1   | id                 | Integer      |         | ✓        | Primary Key, Auto-increment | ID duy nhất của người dùng |
| 2   | github_id          | Integer      |         |          |                             | ID người dùng trên GitHub  |
| 3   | github_username    | String       | 255     | ✓        |                             | Tên đăng nhập trên GitHub  |
| 4   | email              | String       | 255     |          |                             | Email của người dùng       |
| 5   | display_name       | String       | 255     |          |                             | Tên hiển thị               |
| 6   | full_name          | String       | 255     |          |                             | Tên đầy đủ                 |
| 7   | avatar_url         | String       | 500     |          |                             | URL ảnh đại diện           |
| 8   | bio                | Text         |         |          |                             | Tiểu sử                    |
| 9   | location           | String       | 255     |          |                             | Vị trí                     |
| 10  | company            | String       | 255     |          |                             | Công ty                    |
| 11  | blog               | String       | 500     |          |                             | Trang blog                 |
| 12  | twitter_username   | String       | 255     |          |                             | Tên người dùng Twitter     |
| 13  | github_profile_url | String       | 500     |          |                             | URL hồ sơ GitHub           |
| 14  | repos_url          | String       | 500     |          |                             | URL các kho lưu trữ        |
| 15  | is_active          | Boolean      |         |          |                             | Trạng thái hoạt động       |
| 16  | is_verified        | Boolean      |         |          |                             | Trạng thái đã xác minh     |
| 17  | github_created_at  | DateTime     |         |          |                             | Ngày tạo tài khoản GitHub  |
| 18  | last_synced        | DateTime     |         |          |                             | Lần đồng bộ cuối cùng      |
| 19  | created_at         | DateTime     |         |          | `server_default=func.now()` | Ngày tạo                   |
| 20  | updated_at         | DateTime     |         |          | `server_default=func.now()` | Ngày cập nhật              |

## Bảng `repositories`

| TT  | Tên thuộc tính | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị     | Diễn giải                   |
| --- | -------------- | ------------ | ------- | -------- | --------------------------- | --------------------------- |
| 1   | id             | Integer      |         | ✓        | Primary Key, Auto-increment | ID duy nhất của kho lưu trữ |
| 2   | github_id      | Integer      |         | ✓        |                             | ID kho lưu trữ trên GitHub  |
| 3   | owner          | String       | 255     | ✓        |                             | Chủ sở hữu kho lưu trữ      |
| 4   | name           | String       | 255     | ✓        |                             | Tên kho lưu trữ             |
| 5   | full_name      | String       | 500     |          |                             | Tên đầy đủ của kho lưu trữ  |
| 6   | description    | Text         |         |          |                             | Mô tả                       |
| 7   | stars          | Integer      |         |          |                             | Số lượt yêu thích           |
| 8   | forks          | Integer      |         |          |                             | Số lượt fork                |
| 9   | language       | String       | 100     |          |                             | Ngôn ngữ lập trình          |
| 10  | open_issues    | Integer      |         |          |                             | Số lượng issue đang mở      |
| 11  | url            | String       | 500     |          |                             | URL kho lưu trữ             |
| 12  | clone_url      | String       | 500     |          |                             | URL để clone                |
| 13  | is_private     | Boolean      |         |          |                             | Kho lưu trữ riêng tư        |
| 14  | is_fork        | Boolean      |         |          |                             | Là một fork                 |
| 15  | default_branch | String       | 100     |          |                             | Nhánh mặc định              |
| 16  | last_synced    | DateTime     |         |          |                             | Lần đồng bộ cuối cùng       |
| 17  | sync_status    | String       | 20      |          |                             | Trạng thái đồng bộ          |
| 18  | user_id        | Integer      |         |          | `ForeignKey('users.id')`    | ID người dùng sở hữu        |
| 19  | created_at     | DateTime     |         |          | `server_default=func.now()` | Ngày tạo                    |
| 20  | updated_at     | DateTime     |         |          | `server_default=func.now()` | Ngày cập nhật               |

## Bảng `user_repositories`

| TT  | Tên thuộc tính   | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị         | Diễn giải           |
| --- | ---------------- | ------------ | ------- | -------- | ------------------------------- | ------------------- |
| 1   | id               | Integer      |         | ✓        | Primary Key, Auto-increment     | ID duy nhất         |
| 2   | user_id          | Integer      |         | ✓        | `ForeignKey('users.id')`        | ID người dùng       |
| 3   | repository_id    | Integer      |         | ✓        | `ForeignKey('repositories.id')` | ID kho lưu trữ      |
| 4   | role             | String       | 12      | ✓        |                                 | Vai trò             |
| 5   | permissions      | String       | 5       | ✓        |                                 | Quyền hạn           |
| 6   | is_primary_owner | Boolean      |         |          |                                 | Là chủ sở hữu chính |
| 7   | joined_at        | DateTime     |         |          | `server_default=func.now()`     | Ngày tham gia       |
| 8   | last_accessed    | DateTime     |         |          |                                 | Lần truy cập cuối   |
| 9   | created_at       | DateTime     |         |          | `server_default=func.now()`     | Ngày tạo            |
| 10  | updated_at       | DateTime     |         |          | `server_default=func.now()`     | Ngày cập nhật       |

## Bảng `collaborators`

| TT  | Tên thuộc tính  | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị                          | Diễn giải                |
| --- | --------------- | ------------ | ------- | -------- | ------------------------------------------------ | ------------------------ |
| 1   | id              | Integer      |         | ✓        | Primary Key, Auto-increment                      | ID duy nhất              |
| 2   | github_user_id  | Integer      |         | ✓        | `unique=True`                                    | ID người dùng GitHub     |
| 3   | github_username | String       | 255     | ✓        |                                                  | Tên đăng nhập GitHub     |
| 4   | display_name    | String       | 255     |          |                                                  | Tên hiển thị             |
| 5   | email           | String       | 255     |          |                                                  | Email                    |
| 6   | avatar_url      | String       | 500     |          |                                                  | URL ảnh đại diện         |
| 7   | bio             | Text         |         |          |                                                  | Tiểu sử                  |
| 8   | company         | String       | 255     |          |                                                  | Công ty                  |
| 9   | location        | String       | 255     |          |                                                  | Vị trí                   |
| 10  | blog            | String       | 500     |          |                                                  | Trang blog               |
| 11  | is_site_admin   | Boolean      |         |          | `default=False`                                  | Là quản trị viên trang   |
| 12  | node_id         | String       | 255     |          |                                                  | ID node của GitHub       |
| 13  | gravatar_id     | String       | 255     |          |                                                  | ID Gravatar              |
| 14  | type            | String       | 50      |          | `default='User'`                                 | Loại (User/Organization) |
| 15  | user_id         | Integer      |         |          | `ForeignKey('users.id')`                         | Liên kết đến bảng users  |
| 16  | created_at      | DateTime     |         | ✓        | `server_default=func.now()`                      | Ngày tạo                 |
| 17  | updated_at      | DateTime     |         | ✓        | `server_default=func.now(), onupdate=func.now()` | Ngày cập nhật            |

## Bảng `repository_collaborators`

| TT  | Tên thuộc tính    | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị                          | Diễn giải             |
| --- | ----------------- | ------------ | ------- | -------- | ------------------------------------------------ | --------------------- |
| 1   | id                | Integer      |         | ✓        | Primary Key, Auto-increment                      | ID duy nhất           |
| 2   | repository_id     | Integer      |         | ✓        | `ForeignKey('repositories.id')`                  | ID kho lưu trữ        |
| 3   | collaborator_id   | Integer      |         | ✓        | `ForeignKey('collaborators.id')`                 | ID cộng tác viên      |
| 4   | role              | String       | 50      | ✓        |                                                  | Vai trò               |
| 5   | permissions       | String       | 100     |          |                                                  | Quyền hạn             |
| 6   | is_owner          | Boolean      |         | ✓        | `default=False`                                  | Là chủ sở hữu         |
| 7   | joined_at         | DateTime     |         |          |                                                  | Ngày tham gia         |
| 8   | invited_by        | String       | 255     |          |                                                  | Người mời             |
| 9   | invitation_status | String       | 20      |          |                                                  | Trạng thái lời mời    |
| 10  | commits_count     | Integer      |         |          | `default=0`                                      | Số lượng commit       |
| 11  | issues_count      | Integer      |         |          | `default=0`                                      | Số lượng issue        |
| 12  | prs_count         | Integer      |         |          | `default=0`                                      | Số lượng pull request |
| 13  | last_activity     | DateTime     |         |          |                                                  | Hoạt động cuối        |
| 14  | created_at        | DateTime     |         | ✓        | `server_default=func.now()`                      | Ngày tạo              |
| 15  | updated_at        | DateTime     |         | ✓        | `server_default=func.now(), onupdate=func.now()` | Ngày cập nhật         |
| 16  | last_synced       | DateTime     |         | ✓        | `server_default=func.now()`                      | Lần đồng bộ cuối      |

## Bảng `commits`

| TT  | Tên thuộc tính               | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị         | Diễn giải                    |
| --- | ---------------------------- | ------------ | ------- | -------- | ------------------------------- | ---------------------------- |
| 1   | id                           | Integer      |         | ✓        | Primary Key, Auto-increment     | ID duy nhất                  |
| 2   | sha                          | String       | 40      | ✓        |                                 | Mã SHA của commit            |
| 3   | message                      | Text         |         | ✓        |                                 | Thông điệp commit            |
| 4   | author_user_id               | Integer      |         |          | `ForeignKey('users.id')`        | ID người dùng tác giả        |
| 5   | author_name                  | String       | 255     | ✓        |                                 | Tên tác giả                  |
| 6   | author_email                 | String       | 255     | ✓        |                                 | Email tác giả                |
| 7   | committer_user_id            | Integer      |         |          | `ForeignKey('users.id')`        | ID người dùng committer      |
| 8   | committer_name               | String       | 255     |          |                                 | Tên committer                |
| 9   | committer_email              | String       | 255     |          |                                 | Email committer              |
| 10  | repo_id                      | Integer      |         | ✓        | `ForeignKey('repositories.id')` | ID kho lưu trữ               |
| 11  | branch_id                    | Integer      |         |          | `ForeignKey('branches.id')`     | ID nhánh                     |
| 12  | branch_name                  | String       | 255     |          |                                 | Tên nhánh                    |
| 13  | author_role_at_commit        | String       | 20      |          |                                 | Vai trò tác giả lúc commit   |
| 14  | author_permissions_at_commit | String       | 100     |          |                                 | Quyền hạn tác giả lúc commit |
| 15  | date                         | DateTime     |         | ✓        |                                 | Ngày commit                  |
| 16  | committer_date               | DateTime     |         |          |                                 | Ngày committer commit        |
| 17  | insertions                   | Integer      |         |          |                                 | Số dòng code thêm            |
| 18  | deletions                    | Integer      |         |          |                                 | Số dòng code xóa             |
| 19  | files_changed                | Integer      |         |          |                                 | Số file thay đổi             |
| 20  | parent_sha                   | String       | 40      |          |                                 | Mã SHA của commit cha        |
| 21  | is_merge                     | Boolean      |         |          |                                 | Là merge commit              |
| 22  | merge_from_branch            | String       | 255     |          |                                 | Nhánh được merge từ          |
| 23  | modified_files               | JSON         |         |          |                                 | Danh sách file bị sửa đổi    |
| 24  | file_types                   | JSON         |         |          |                                 | Thống kê loại file           |
| 25  | modified_directories         | JSON         |         |          |                                 | Thống kê thư mục bị sửa đổi  |
| 26  | total_changes                | Integer      |         |          |                                 | Tổng số thay đổi             |
| 27  | change_type                  | String       | 50      |          |                                 | Loại thay đổi                |
| 28  | commit_size                  | String       | 20      |          |                                 | Kích thước commit            |
| 29  | created_at                   | DateTime     |         |          | `server_default=func.now()`     | Ngày tạo                     |
| 30  | last_synced                  | DateTime     |         |          | `server_default=func.now()`     | Lần đồng bộ cuối             |
| 31  | diff_content                 | Text         |         |          |                                 | Nội dung diff                |

## Bảng `branches`

| TT  | Tên thuộc tính         | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị         | Diễn giải               |
| --- | ---------------------- | ------------ | ------- | -------- | ------------------------------- | ----------------------- |
| 1   | id                     | Integer      |         | ✓        | Primary Key, Auto-increment     | ID duy nhất             |
| 2   | name                   | String       | 255     | ✓        |                                 | Tên nhánh               |
| 3   | repo_id                | Integer      |         | ✓        | `ForeignKey('repositories.id')` | ID kho lưu trữ          |
| 4   | creator_user_id        | Integer      |         |          | `ForeignKey('users.id')`        | ID người tạo            |
| 5   | creator_name           | String       | 255     |          |                                 | Tên người tạo           |
| 6   | last_committer_user_id | Integer      |         |          | `ForeignKey('users.id')`        | ID người commit cuối    |
| 7   | last_committer_name    | String       | 255     |          |                                 | Tên người commit cuối   |
| 8   | sha                    | String       | 40      |          |                                 | Mã SHA của commit cuối  |
| 9   | is_default             | Boolean      |         |          |                                 | Là nhánh mặc định       |
| 10  | is_protected           | Boolean      |         |          |                                 | Nhánh được bảo vệ       |
| 11  | created_at             | DateTime     |         |          |                                 | Ngày tạo                |
| 12  | last_commit_date       | DateTime     |         |          |                                 | Ngày commit cuối        |
| 13  | last_synced            | DateTime     |         |          | `server_default=func.now()`     | Lần đồng bộ cuối        |
| 14  | commits_count          | Integer      |         |          |                                 | Số lượng commit         |
| 15  | contributors_count     | Integer      |         |          |                                 | Số lượng người đóng góp |

## Bảng `pull_requests`

| TT  | Tên thuộc tính | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị         | Diễn giải                   |
| --- | -------------- | ------------ | ------- | -------- | ------------------------------- | --------------------------- |
| 1   | id             | Integer      |         | ✓        | Primary Key, Auto-increment     | ID duy nhất                 |
| 2   | github_id      | Integer      |         |          |                                 | ID pull request trên GitHub |
| 3   | title          | String       | 255     | ✓        |                                 | Tiêu đề                     |
| 4   | description    | String       | 255     |          |                                 | Mô tả                       |
| 5   | state          | String       | 50      |          |                                 | Trạng thái                  |
| 6   | repo_id        | Integer      |         | ✓        | `ForeignKey('repositories.id')` | ID kho lưu trữ              |
| 7   | created_at     | DateTime     |         |          | `server_default=func.now()`     | Ngày tạo                    |
| 8   | updated_at     | DateTime     |         |          | `server_default=func.now()`     | Ngày cập nhật               |

## Bảng `issues`

| TT  | Tên thuộc tính | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị         | Diễn giải            |
| --- | -------------- | ------------ | ------- | -------- | ------------------------------- | -------------------- |
| 1   | id             | Integer      |         | ✓        | Primary Key, Auto-increment     | ID duy nhất          |
| 2   | github_id      | Integer      |         |          |                                 | ID issue trên GitHub |
| 3   | title          | String       | 255     | ✓        |                                 | Tiêu đề              |
| 4   | body           | Text         |         |          |                                 | Nội dung             |
| 5   | state          | String       | 50      | ✓        |                                 | Trạng thái           |
| 6   | created_at     | DateTime     |         | ✓        |                                 | Ngày tạo             |
| 7   | updated_at     | DateTime     |         |          |                                 | Ngày cập nhật        |
| 8   | repo_id        | Integer      |         | ✓        | `ForeignKey('repositories.id')` | ID kho lưu trữ       |

## Bảng `assignments`

| TT  | Tên thuộc tính | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị     | Diễn giải          |
| --- | -------------- | ------------ | ------- | -------- | --------------------------- | ------------------ |
| 1   | id             | Integer      |         | ✓        | Primary Key, Auto-increment | ID duy nhất        |
| 2   | task_name      | String       | 255     | ✓        |                             | Tên công việc      |
| 3   | description    | String       | 255     |          |                             | Mô tả              |
| 4   | is_completed   | Boolean      |         |          |                             | Đã hoàn thành      |
| 5   | user_id        | Integer      |         | ✓        | `ForeignKey('users.id')`    | ID người được giao |
| 6   | created_at     | DateTime     |         |          | `server_default=func.now()` | Ngày tạo           |
| 7   | updated_at     | DateTime     |         |          | `server_default=func.now()` | Ngày cập nhật      |

## Bảng `project_tasks`

| TT  | Tên thuộc tính           | Kiểu dữ liệu | Độ rộng | Not null | Ràng Buộc/ Miền giá trị         | Diễn giải                            |
| --- | ------------------------ | ------------ | ------- | -------- | ------------------------------- | ------------------------------------ |
| 1   | id                       | Integer      |         | ✓        | Primary Key, Auto-increment     | ID duy nhất                          |
| 2   | title                    | String       | 255     | ✓        |                                 | Tiêu đề                              |
| 3   | description              | Text         |         |          |                                 | Mô tả                                |
| 4   | assignee_user_id         | Integer      |         |          | `ForeignKey('users.id')`        | ID người được giao                   |
| 5   | assignee_github_username | String       | 100     |          |                                 | Tên đăng nhập GitHub người được giao |
| 6   | status                   | String       | 11      | ✓        | `Enum(TaskStatus)`              | Trạng thái                           |
| 7   | priority                 | String       | 6       | ✓        | `Enum(TaskPriority)`            | Độ ưu tiên                           |
| 8   | due_date                 | String       | 10      |          |                                 | Ngày hết hạn                         |
| 9   | repository_id            | Integer      |         |          | `ForeignKey('repositories.id')` | ID kho lưu trữ                       |
| 10  | repo_owner               | String       | 100     |          |                                 | Chủ sở hữu kho lưu trữ               |
| 11  | repo_name                | String       | 100     |          |                                 | Tên kho lưu trữ                      |
| 12  | is_completed             | Boolean      |         |          |                                 | Đã hoàn thành                        |
| 13  | created_at               | DateTime     |         |          | `server_default=func.now()`     | Ngày tạo                             |
| 14  | updated_at               | DateTime     |         |          | `server_default=func.now()`     | Ngày cập nhật                        |
| 15  | created_by_user_id       | Integer      |         |          | `ForeignKey('users.id')`        | ID người tạo                         |
| 16  | created_by               | String       | 100     |          |                                 | Tên người tạo                        |
