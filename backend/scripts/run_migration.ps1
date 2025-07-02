# PowerShell script to run enhanced commit fields migration
# run_migration.ps1

Write-Host "🚀 Enhanced Commit Fields Migration" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Change to backend directory
$BackendDir = Split-Path -Parent $PSScriptRoot
Set-Location $BackendDir

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  No virtual environment found. Continuing with system Python..." -ForegroundColor Yellow
}

# Method 1: Try Python migration script
Write-Host "Method 1: Running Python migration script..." -ForegroundColor Cyan
try {
    python scripts\run_enhanced_migration.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Migration completed successfully!" -ForegroundColor Green
        exit 0
    }
} catch {
    Write-Host "❌ Python migration script failed: $_" -ForegroundColor Red
}

# Method 2: Try Alembic directly
Write-Host "Method 2: Running Alembic directly..." -ForegroundColor Cyan
try {
    alembic upgrade head
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Alembic migration completed successfully!" -ForegroundColor Green
        exit 0
    }
} catch {
    Write-Host "❌ Alembic migration failed: $_" -ForegroundColor Red
}

# Method 3: Manual SQL approach
Write-Host "Method 3: Manual SQL approach needed..." -ForegroundColor Yellow
Write-Host "Please run the SQL commands in scripts\manual_add_enhanced_fields.sql manually in your database." -ForegroundColor Yellow
Write-Host "Or use your database management tool to add these columns to the 'commits' table:" -ForegroundColor Yellow
Write-Host "  - modified_files (TEXT/JSON)" -ForegroundColor White
Write-Host "  - file_types (TEXT/JSON)" -ForegroundColor White
Write-Host "  - modified_directories (TEXT/JSON)" -ForegroundColor White
Write-Host "  - total_changes (INTEGER)" -ForegroundColor White
Write-Host "  - change_type (VARCHAR(50))" -ForegroundColor White
Write-Host "  - commit_size (VARCHAR(20))" -ForegroundColor White

Write-Host "After adding columns manually, restart your FastAPI application." -ForegroundColor Yellow

exit 1
