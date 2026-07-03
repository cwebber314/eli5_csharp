$ollama = (Get-Process ollama -EA SilentlyContinue | Select -First 1).Path

Write-Host "===== GPU (whole card) ====="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

Write-Host "`n===== Ollama model (VRAM + CPU/GPU split) ====="
if ($ollama) { & $ollama ps } else { "ollama server not running" }

Write-Host "`n===== Process RAM / CPU ====="
Get-Process ollama*, *llama*, python* -EA SilentlyContinue |
  Select Name, Id, @{n='RAM_GB';e={[math]::Round($_.WorkingSet64/1GB,2)}}, @{n='CPU_s';e={[math]::Round($_.CPU,1)}} |
  Format-Table -AutoSize

Write-Host "===== Disk ====="
foreach ($d in 'models','chroma_db','.venv','repos') { if (Test-Path $d) {
  $s=(Get-ChildItem $d -Recurse -File -EA SilentlyContinue | Measure Length -Sum).Sum
  "{0,-11} {1,8:N2} GB" -f $d, ($s/1GB) } }