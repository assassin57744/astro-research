Get-ChildItem -Recurse -Exclude "venv", ".venv", ".git", "__pycache__" | ForEach-Object {
    # 计算当前文件/文件夹的深度
    $relativeSubPath = $_.FullName.Substring($PWD.FullName.Length)
    $levels = $relativeSubPath.Split([System.IO.Path]::DirectorySeparatorChar, [System.StringSplitOptions]::RemoveEmptyEntries)
    $depth = $levels.Count - 1
    
    # 根据深度生成缩进和连接符
    $indent = "   " * $depth
    $connector = if ($depth -gt 0) { "└── " } else { "├── " }
    
    # 区分文件夹和文件，并赋予不同的视觉图标
    if ($_.PSIsContainer) {
        Write-Host "${indent}${connector}📁 $($_.Name)" -ForegroundColor Cyan
    } else {
        Write-Host "${indent}${connector}📄 $($_.Name)"
    }
}
