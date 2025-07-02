# Fix: Database Columns Missing Error

## Problem
The error `column commits.modified_files does not exist` indicates that the database table doesn't have the new enhanced fields that were added to the model.

## Solution Options

### Option 1: Run Python Migration Script (Recommended)

```bash
cd backend
python scripts/add_enhanced_fields_direct.py
```

This script will:
- Check which columns are missing
- Add only the missing columns
- Verify the columns were added successfully

### Option 2: Run Alembic Migration

```bash
cd backend
python scripts/run_enhanced_migration.py
```

Or directly:
```bash
cd backend
alembic upgrade head
```

### Option 3: PowerShell Script (Windows)

```powershell
cd backend
.\scripts\run_migration.ps1
```

### Option 4: Manual SQL (Last Resort)

If all automated methods fail, run these SQL commands manually in your database:

```sql
ALTER TABLE commits ADD COLUMN modified_files TEXT;
ALTER TABLE commits ADD COLUMN file_types TEXT;
ALTER TABLE commits ADD COLUMN modified_directories TEXT;
ALTER TABLE commits ADD COLUMN total_changes INTEGER;
ALTER TABLE commits ADD COLUMN change_type VARCHAR(50);
ALTER TABLE commits ADD COLUMN commit_size VARCHAR(20);
```

## Verification

After running any of the above methods, verify the columns exist:

### PostgreSQL
```sql
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'commits' 
AND column_name IN ('modified_files', 'file_types', 'modified_directories', 'total_changes', 'change_type', 'commit_size');
```

### SQLite
```sql
PRAGMA table_info(commits);
```

You should see the new columns in the output.

## What These Fields Do

- **modified_files**: JSON array of file paths that were changed
- **file_types**: JSON object with file extensions and their counts
- **modified_directories**: JSON object with directories and change counts  
- **total_changes**: Integer - total additions + deletions
- **change_type**: String - feature, bugfix, refactor, etc.
- **commit_size**: String - small, medium, large, massive

## After Migration

1. **Restart your FastAPI application**
2. **Test the endpoints** - the error should be gone
3. **Sync some commits** to populate the new fields:
   ```http
   POST /api/github/{owner}/{repo}/sync-enhanced
   ```

## Troubleshooting

### If migration fails:
1. Check database connection in your `.env` file
2. Ensure database user has ALTER TABLE permissions
3. Try the manual SQL approach
4. Check logs for specific error messages

### If columns exist but still get errors:
1. Restart the FastAPI application
2. Clear any database connection pools
3. Check if you're using the correct database

### If you're using Docker:
```bash
# Enter the container
docker exec -it your-container-name bash

# Run migration inside container
cd /app/backend
python scripts/add_enhanced_fields_direct.py
```

## Files Created for This Fix

1. `scripts/add_enhanced_fields_direct.py` - Direct SQL approach
2. `scripts/run_enhanced_migration.py` - Alembic migration runner
3. `scripts/run_migration.ps1` - PowerShell script for Windows
4. `scripts/manual_add_enhanced_fields.sql` - Manual SQL commands
5. `migrations/versions/add_enhanced_commit_fields.py` - Updated migration file

Choose the method that works best for your setup!
