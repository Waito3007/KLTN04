
Table users {
  id int [pk, increment]
  github_id int
  github_username varchar(255) [not null]
  email varchar(255)
  display_name varchar(255)
  full_name varchar(255)
  avatar_url varchar(500)
  bio text
  location varchar(255)
  company varchar(255)
  blog varchar(500)
  twitter_username varchar(255)
  github_profile_url varchar(500)
  repos_url varchar(500)
  is_active boolean
  is_verified boolean
  github_created_at datetime
  last_synced datetime
  created_at datetime [default: `now()`]
  updated_at datetime [default: `now()`]
}

Table repositories {
  id int [pk, increment]
  github_id int [not null]
  owner varchar(255) [not null]
  name varchar(255) [not null]
  full_name varchar(500)
  description text
  stars int
  forks int
  language varchar(100)
  open_issues int
  url varchar(500)
  clone_url varchar(500)
  is_private boolean
  is_fork boolean
  default_branch varchar(100)
  last_synced datetime
  sync_status varchar(20)
  user_id int [ref: > users.id]
  created_at datetime [default: `now()`]
  updated_at datetime [default: `now()`]
}

Table branches {
  id int [pk, increment]
  name varchar(255) [not null]
  repo_id int [not null, ref: > repositories.id]
  creator_user_id int [ref: > users.id]
  creator_name varchar(255)
  last_committer_user_id int [ref: > users.id]
  last_committer_name varchar(255)
  sha varchar(40)
  is_default boolean
  is_protected boolean
  created_at datetime
  last_commit_date datetime
  last_synced datetime [default: `now()`]
  commits_count int
  contributors_count int
}

Table commits {
  id int [pk, increment]
  sha varchar(40) [not null]
  message text [not null]
  author_user_id int [ref: > users.id]
  author_name varchar(255) [not null]
  author_email varchar(255) [not null]
  committer_user_id int [ref: > users.id]
  committer_name varchar(255)
  committer_email varchar(255)
  repo_id int [not null, ref: > repositories.id]
  branch_id int [ref: > branches.id]
  branch_name varchar(255)
  author_role_at_commit varchar(20)
  author_permissions_at_commit varchar(100)
  date datetime [not null]
  committer_date datetime
  insertions int
  deletions int
  files_changed int
  parent_sha varchar(40)
  is_merge boolean
  merge_from_branch varchar(255)
  modified_files json 
  file_types json 
  modified_directories json 
  total_changes int
  change_type varchar(50)
  commit_size varchar(20)
  created_at datetime [default: `now()`]
  last_synced datetime [default: `now()`]
}

Table collaborators {
  id int [pk, increment]
  github_user_id int [unique, not null]
  github_username varchar(255) [not null]
  display_name varchar(255)
  email varchar(255)
  avatar_url varchar(500)
  bio text
  company varchar(255)
  location varchar(255)
  blog varchar(500)
  is_site_admin boolean [default: false]
  node_id varchar(255)
  gravatar_id varchar(255)
  type varchar(50) [default: 'User']
  user_id int [ref: > users.id]
  created_at datetime [not null, default: `now()`]
  updated_at datetime [not null, default: `now()`, on update: `now()`]
}

Table repository_collaborators {
  id int [pk, increment]
  repository_id int [not null, ref: > repositories.id]
  collaborator_id int [not null, ref: > collaborators.id]
  role varchar(50) [not null]
  permissions varchar(100)
  is_owner boolean [not null, default: false]
  joined_at datetime
  invited_by varchar(255)
  invitation_status varchar(20)
  commits_count int [default: 0]
  issues_count int [default: 0]
  prs_count int [default: 0]
  last_activity datetime
  created_at datetime [not null, default: `now()`]
  updated_at datetime [not null, default: `now()`, on update: `now()`]
  last_synced datetime [not null, default: `now()`]

  indexes {
    (repository_id, collaborator_id) [unique]
  }
}

Table issues {
  id int [pk, increment]
  github_id int
  title varchar(255) [not null]
  body text
  state varchar(50) [not null]
  created_at datetime [not null]
  updated_at datetime
  repo_id int [not null, ref: > repositories.id]
}

Table pull_requests {
  id int [pk, increment]
  github_id int
  title varchar(255) [not null]
  description varchar(255)
  state varchar(50)
  repo_id int [not null, ref: > repositories.id]
  created_at datetime [default: `now()`]
  updated_at datetime [default: `now()`]
}

Table user_repositories {
  id int [pk, increment]
  user_id int [not null, ref: > users.id]
  repository_id int [not null, ref: > repositories.id]
  role varchar(12) [not null]
  permissions varchar(5) [not null]
  is_primary_owner boolean
  joined_at datetime [default: `now()`]
  last_accessed datetime
  created_at datetime [default: `now()`]
  updated_at datetime [default: `now()`]
}

Table assignments {
  id int [pk, increment]
  task_name varchar(255) [not null]
  description varchar(255)
  is_completed boolean
  user_id int [not null, ref: > users.id]
  created_at datetime [default: `now()`]
  updated_at datetime [default: `now()`]
}

Table project_tasks {
  id int [pk, increment]
  title varchar(255) [not null]
  description text
  assignee_user_id int [ref: > users.id]
  assignee_github_username varchar(100)
  status varchar(11) [not null]
  priority varchar(6) [not null]
  due_date varchar(10)
  repository_id int [ref: > repositories.id]
  repo_owner varchar(100)
  repo_name varchar(100)
  is_completed boolean
  created_at datetime [default: `now()`]
  updated_at datetime [default: `now()`]
  created_by_user_id int [ref: > users.id]
  created_by varchar(100)
}