# PowerShell script to clean Git history and push to GitHub
# This removes the large VA6_result.csv file from Git history

Write-Host "🧹 Cleaning Git repository..." -ForegroundColor Cyan

# Step 1: Remove .git folders from subdirectories (convert submodules to regular folders)
Write-Host "`n1️⃣ Converting submodules to regular folders..." -ForegroundColor Yellow
Remove-Item -Recurse -Force "interva6\.git" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "vman3\.git" -ErrorAction SilentlyContinue
Remove-Item -Force ".gitmodules" -ErrorAction SilentlyContinue
Write-Host "   ✓ Submodules converted" -ForegroundColor Green

# Step 2: Stage all current files
Write-Host "`n2️⃣ Staging all files..." -ForegroundColor Yellow
git add .
git status --short
Write-Host "   ✓ Files staged" -ForegroundColor Green

# Step 3: Commit current state
Write-Host "`n3️⃣ Committing changes..." -ForegroundColor Yellow
git commit -m "Add Docker deployment configuration and clean repository"
Write-Host "   ✓ Changes committed" -ForegroundColor Green

# Step 4: Remove large file from ALL commits in history
Write-Host "`n4️⃣ Removing large file from Git history..." -ForegroundColor Yellow
Write-Host "   This may take a minute..." -ForegroundColor Gray
$env:FILTER_BRANCH_SQUELCH_WARNING = "1"
git filter-branch --force --index-filter `
    "git rm --cached --ignore-unmatch VA_output/VA6_result.csv VA_output/VA5_result.csv" `
    --prune-empty --tag-name-filter cat -- --all

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ Large files removed from history" -ForegroundColor Green
} else {
    Write-Host "   ⚠ Could not remove file (might already be removed)" -ForegroundColor Yellow
}

# Step 5: Force garbage collection to reduce repo size
Write-Host "`n5️⃣ Cleaning up repository..." -ForegroundColor Yellow
git reflog expire --expire=now --all
git gc --prune=now --aggressive
Write-Host "   ✓ Repository cleaned" -ForegroundColor Green

# Step 6: Force push to GitHub
Write-Host "`n6️⃣ Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "   ⚠ This will FORCE PUSH to overwrite remote history" -ForegroundColor Red
$confirm = Read-Host "   Continue? (yes/no)"

if ($confirm -eq "yes") {
    git push -u origin main --force
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ SUCCESS! Repository pushed to GitHub!" -ForegroundColor Green
        Write-Host "   Your code is now live at: https://github.com/shukran-yega/GUI-interVA-5-6" -ForegroundColor Cyan
    } else {
        Write-Host "`n❌ Push failed. Check the error message above." -ForegroundColor Red
    }
} else {
    Write-Host "`n⏸ Push cancelled. Run 'git push -u origin main --force' when ready." -ForegroundColor Yellow
}

Write-Host "`n✨ Done!" -ForegroundColor Cyan
