import requests
import json
import os

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000/api"
OWNER = "Waito3007"
REPO = "KLTN04"
BRANCH = "NghiaDemo"
MAX_COMMITS_TO_SYNC = 5

# --- GitHub Token ---
github_token = "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" 

if github_token == "":
    print("ERROR: Please replace 'YOUR_GITHUB_TOKEN_HERE' with your actual GitHub API token in the script file.")
    exit()

headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/json"
}

# --- Step 1: Run sync ---
print(f"Starting commit sync for {OWNER}/{REPO}:{BRANCH} with diffs...")
sync_url = f"{BASE_URL}/github/{OWNER}/{REPO}/branches/{BRANCH}/sync-commits"
sync_params = {
    "max_pages": 1,
    "per_page": MAX_COMMITS_TO_SYNC,
    "include_diff": True,
    "force_update": True
}

try:
    sync_response = requests.post(sync_url, headers=headers, params=sync_params, timeout=300)
    sync_response.raise_for_status()
    sync_result = sync_response.json()
    print("Sync API call successful.")
    print(json.dumps(sync_result, indent=2))

    if sync_result.get("stats", {}).get("total_fetched_from_github", 0) == 0:
        print("Sync completed but fetched 0 commits. Cannot verify diff content.")
        exit()

except requests.exceptions.RequestException as e:
    print(f"Error during sync API call: {e}")
    if e.response:
        print(f"Response body: {e.response.text}")
    exit()

# --- Step 2: Verify ---
print(f"\nVerifying by fetching commits from the database...")
verify_url = f"{BASE_URL}/commits/{OWNER}/{REPO}/branches/{BRANCH}/commits"
verify_params = {
    "limit": MAX_COMMITS_TO_SYNC
}

try:
    verify_response = requests.get(verify_url, params=verify_params, timeout=60)
    verify_response.raise_for_status()
    verify_result = verify_response.json()
    
    commits = verify_result.get("commits", [])
    if not commits:
        print("Verification failed: No commits were found in the database after sync.")
        exit()

    # --- Step 3: Check for diff_content ---
    diff_found = False
    for commit in commits:
        if commit.get("diff_content") and isinstance(commit["diff_content"], str) and len(commit["diff_content"]) > 0:
            print(f"SUCCESS: Found diff_content for commit {commit['sha'][:7]}")
            diff_found = True
            break

    if diff_found:
        print("\nTest Passed: The 'diff_content' field is being correctly synced and saved.")
    else:
        print("\nTest Failed: No commits found with populated 'diff_content' after sync.")

except requests.exceptions.RequestException as e:
    print(f"Error during verification API call: {e}")
    if e.response:
        print(f"Response body: {e.response.text}")
    exit()