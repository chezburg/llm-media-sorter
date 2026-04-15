# --- Configuration ---
$TargetFolders = @("./keep", "./review", "./trash")
$DryRun = $false

Write-Host "Starting Hash-Based Deduplication for 5,000 files..." -ForegroundColor Cyan

# 1. Gather all files first
$AllFiles = Get-ChildItem -Path $TargetFolders -File -Recurse
$TotalCount = $AllFiles.Count
Write-Host "[*] Found $TotalCount files. Calculating hashes..." -ForegroundColor Yellow

$Hashes = @()
$Current = 0

# 2. Calculate Hashes with a Progress Bar
foreach ($File in $AllFiles) {
    $Current++
    $Percent = ($Current / $TotalCount) * 100
    Write-Progress -Activity "Analyzing Files" -Status "Hashing: $($File.Name)" -PercentComplete $Percent
    
    try {
        $FileHash = Get-FileHash -Path $File.FullName -Algorithm SHA256
        $Hashes += [PSCustomObject]@{
            Path = $File.FullName
            Name = $File.Name
            Hash = $FileHash.Hash
        }
    } catch {
        Write-Host "[!] Could not process: $($File.Name)" -ForegroundColor Red
    }
}

# 3. Group by Hash and identify duplicates
$Groups = $Hashes | Group-Object Hash | Where-Object { $_.Count -gt 1 }

if ($Groups.Count -eq 0) {
    Write-Host "No duplicate files found based on content." -ForegroundColor Green
} else {
    foreach ($Group in $Groups) {
        # Logic: Keep the file with the shortest name (usually the one without " - Copy")
        $Sorted = $Group.Group | Sort-Object { $_.Name.Length }
        $Keep = $Sorted[0]
        $Duplicates = $Sorted | Select-Object -Skip 1

        Write-Host "`n[Match Found] Content is identical for $($Keep.Name):" -ForegroundColor Green
        
        foreach ($Dup in $Duplicates) {
            if ($DryRun) {
                Write-Host "    [DRY RUN] Would delete: $($Dup.Path)" -ForegroundColor Gray
            } else {
                Write-Host "    [DELETE] Removing: $($Dup.Path)" -ForegroundColor Red
                Remove-Item -Path $Dup.Path -Force
            }
        }
    }
}

Write-Host "`n--------------------------------" -ForegroundColor Cyan
if ($DryRun) { Write-Host "Analysis complete. Set `$DryRun = `$false` and run again to delete." }
else { Write-Host "Cleanup complete!" }