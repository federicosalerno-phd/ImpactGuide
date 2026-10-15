<#
.SYNOPSIS
  The launcher behind the ImpactGuide icon on the desktop.

  install_shortcut.ps1 copies it to %LOCALAPPDATA%\ImpactGuide\launch, on the
  local disk, because the project may sit on Google Drive, whose letter does
  not exist for the first minutes after a login: a shortcut pointing into it
  would do nothing at all when clicked too soon. So the click never gets lost:
    - the launcher is local, so it always runs;
    - if the project is not readable yet it starts Google Drive if needed and
      waits for it in a small window, looking on every drive letter;
    - it finds an interpreter that can open the window, and remembers it;
    - it does not open a second window: it brings the open one to the front;
    - if the app dies on startup it shows the error.

  Checks (they do not start the app):
    powershell -File launcher.ps1 -Diag
    powershell -File launcher.ps1 -WhichPython
#>
param(
    [string]$App = "",         # impactguide_app.py to open; without it, target.json
    [int]$WaitSec = 900,       # how long to wait for the drive, at most
    [switch]$Diag,             # say what is visible from here, and stop
    [switch]$WhichPython       # print the interpreter it would use, and stop
)

$ErrorActionPreference = 'Stop'

# LAUNCHER-VERSION: 2
# install_shortcut.ps1 reads the number above: when it changes, the local copy is rewritten.

$StateDir = Join-Path $env:LOCALAPPDATA 'ImpactGuide\launch'
$LogFile  = Join-Path $StateDir 'launch.log'
$ErrFile  = Join-Path $StateDir 'last-error.txt'
$PyFile   = Join-Path $StateDir 'python.txt'

function Write-Log([string]$line) {
    try {
        if (-not (Test-Path -LiteralPath $StateDir)) { New-Item -ItemType Directory -Path $StateDir -Force | Out-Null }
        Add-Content -LiteralPath $LogFile -Value ("{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $line) -Encoding UTF8
        if ((Get-Item -LiteralPath $LogFile).Length -gt 100KB) {
            Set-Content -LiteralPath $LogFile -Value (Get-Content -LiteralPath $LogFile -Tail 300) -Encoding UTF8
        }
    } catch { }
}

# --- where the app is ---------------------------------------------------------
function Get-Target {
    $o = $null
    $f = Join-Path $StateDir 'target.json'
    if (Test-Path -LiteralPath $f) { try { $o = Get-Content -LiteralPath $f -Raw -Encoding UTF8 | ConvertFrom-Json } catch { } }
    $script = if ($App) { $App } elseif ($o) { "$($o.app)" } else { "" }
    $rel = if ($o) { "$($o.rel)" } else { "" }
    if (-not $rel -and $script -match '^[A-Za-z]:[\\/](.+)$') { $rel = $matches[1] }
    return @{ app = $script; rel = $rel }
}

# On Drive a file can exist as a placeholder before it can be read: only a real read counts.
function Test-Readable([string]$path) {
    if (-not $path) { return $false }
    try {
        if (-not [IO.File]::Exists($path)) { return $false }
        $fs = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try { $b = New-Object byte[] 64; return ($fs.Read($b, 0, 64) -gt 0) } finally { $fs.Dispose() }
    } catch { return $false }
}

# The saved path first, then the same path under every drive letter, since a
# synced drive is not promised the same letter at every login.
function Find-App($target) {
    if (Test-Readable $target.app) { return $target.app }
    if (-not $target.rel) { return $null }
    foreach ($d in [IO.DriveInfo]::GetDrives()) {
        try { if (-not $d.IsReady) { continue } } catch { continue }
        $p = Join-Path $d.Name $target.rel
        if (Test-Readable $p) { return $p }
    }
    return $null
}

function Start-GoogleDrive {
    try { if (Get-Process -Name GoogleDriveFS -ErrorAction SilentlyContinue) { return 'running' } } catch { }
    $exe = $null
    try {
        $run = Get-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run' -ErrorAction SilentlyContinue
        if ($run -and "$($run.GoogleDriveFS)" -match '"?([A-Za-z]:\\[^"]*GoogleDriveFS\.exe)"?') { $exe = $matches[1] }
    } catch { }
    if (-not $exe -or -not (Test-Path -LiteralPath $exe)) { return 'not found' }
    try { Start-Process -FilePath $exe -ArgumentList '--startup_mode' -WindowStyle Hidden; return "started" }
    catch { return "could not start: $($_.Exception.Message)" }
}

# --- the interpreter ------------------------------------------------------------
function Get-PythonCandidates([string]$appPath) {
    $out = @()
    if ($env:IMPACTGUIDE_PYTHON) { $out += $env:IMPACTGUIDE_PYTHON }
    $out += 'C:\envs\marker-impact\Scripts\python.exe'       # the environment of 04_code/requirements.txt
    if ($appPath) { $out += (Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $appPath))) '.venv\Scripts\python.exe') }
    return ($out | Where-Object { $_ } | Select-Object -Unique)
}

function Test-Python([string]$py) {
    if (-not $py -or -not (Test-Path -LiteralPath $py)) { return $false }
    try {
        $p = Start-Process -FilePath $py -ArgumentList '-c', '"import slantui, cv2, PyQt6.QtWebEngineWidgets"' -WindowStyle Hidden -PassThru
        if (-not $p.WaitForExit(60000)) { try { $p.Kill() } catch { }; return $false }
        return ($p.ExitCode -eq 0)
    } catch { return $false }
}

function Find-Python([string]$appPath) {
    if (Test-Path -LiteralPath $PyFile) {
        try { $c = (Get-Content -LiteralPath $PyFile -Raw -Encoding UTF8).Trim(); if ($c -and (Test-Path -LiteralPath $c)) { return $c } } catch { }
    }
    foreach ($cand in (Get-PythonCandidates $appPath)) {
        if (Test-Python $cand) {
            try { Set-Content -LiteralPath $PyFile -Value $cand -Encoding UTF8 } catch { }
            Write-Log "interpreter: $cand"
            return $cand
        }
    }
    return $null
}

# --- one window only --------------------------------------------------------------
function Get-RunningApp {
    try {
        $q = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
             Where-Object { "$($_.CommandLine)" -like '*impactguide_app.py*' }
        foreach ($r in $q) {
            $p = Get-Process -Id ([int]$r.ProcessId) -ErrorAction SilentlyContinue
            if ($p -and $p.MainWindowHandle -ne [IntPtr]::Zero) { return [int]$r.ProcessId }
        }
        if ($q) { return [int]@($q)[0].ProcessId }
    } catch { }
    return 0
}

function Show-Existing([int]$processId) {
    try {
        Add-Type -Namespace ImpactGuide -Name Fore -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
[DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int n);
[DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
'@ -ErrorAction SilentlyContinue
        $p = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if (-not $p -or $p.MainWindowHandle -eq 0) { return $false }
        if ([ImpactGuide.Fore]::IsIconic($p.MainWindowHandle)) { [void][ImpactGuide.Fore]::ShowWindowAsync($p.MainWindowHandle, 9) }
        [void][ImpactGuide.Fore]::SetForegroundWindow($p.MainWindowHandle)
        return $true
    } catch { return $false }
}

# --- the small window: waiting for the drive, or an error ---------------------------
# No colour is written here: theme.json holds the SlantUI roles of the app's
# palette (launcher_assets.py), and without it the window borrows the system's.
function Get-Theme {
    $keys = @('surface', 'raised', 'line', 'control', 'controlHover', 'text', 'muted', 'accent', 'error')
    $t = @{}
    $f = Join-Path $StateDir 'theme.json'
    if (Test-Path -LiteralPath $f) {
        try {
            $o = Get-Content -LiteralPath $f -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($k in $keys) {
                $v = "$($o.$k)".Trim()
                if ($v -match '^#[0-9A-Fa-f]{8}$') { $v = '#' + $v.Substring(7, 2) + $v.Substring(1, 6) }
                if ($v -match '^#[0-9A-Fa-f]{6,8}$') { $t[$k] = $v }
            }
        } catch { }
    }
    if ($t.Count -eq $keys.Count) { return $t }
    Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
    $sys = @{ surface = 'WindowColor'; raised = 'ControlColor'; line = 'ActiveBorderColor'; control = 'ControlColor'
              controlHover = 'ControlLightColor'; text = 'WindowTextColor'; muted = 'GrayTextColor'
              accent = 'HighlightColor'; error = 'WindowTextColor' }
    foreach ($k in $keys) {
        if (-not $t.ContainsKey($k)) {
            $c = [Windows.SystemColors]::($sys[$k])
            $t[$k] = "#{0:X2}{1:X2}{2:X2}" -f $c.R, $c.G, $c.B
        }
    }
    return $t
}

function Show-Box([string]$head, [string]$text, [bool]$isError, $shared) {
    try {
        Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
        $t = Get-Theme
        $fg = if ($isError) { $t.error } else { $t.muted }
        $xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        WindowStyle="None" AllowsTransparency="True" Background="Transparent" ResizeMode="NoResize" ShowInTaskbar="True"
        Topmost="True" WindowStartupLocation="CenterScreen" SizeToContent="Height" Width="440" Title="ImpactGuide" FontFamily="Segoe UI">
  <Border Background="$($t.raised)" CornerRadius="10" BorderBrush="$($t.line)" BorderThickness="1" Padding="18,16,18,14">
    <StackPanel>
      <StackPanel Orientation="Horizontal">
        <Image x:Name="Logo" Width="30" Height="30" VerticalAlignment="Top" Margin="0,1,12,0"/>
        <StackPanel Width="360">
          <TextBlock Text="$head" Foreground="$($t.text)" FontSize="15" FontWeight="SemiBold"/>
          <TextBlock x:Name="Sub" Foreground="$fg" FontSize="12" Margin="0,3,0,0" TextWrapping="Wrap"/>
        </StackPanel>
      </StackPanel>
      <Button x:Name="Main" Content="Close" HorizontalAlignment="Right" Margin="0,12,0,0" Padding="14,4,14,4"
              Background="$($t.control)" Foreground="$($t.text)" BorderThickness="0"/>
    </StackPanel>
  </Border>
</Window>
"@
        $w = [Windows.Markup.XamlReader]::Parse($xaml)
        $w.FindName('Sub').Text = $text
        $ico = Join-Path $StateDir 'icon.ico'
        if (Test-Path -LiteralPath $ico) {
            $img = New-Object Windows.Media.Imaging.BitmapImage
            $img.BeginInit(); $img.UriSource = New-Object Uri($ico); $img.DecodePixelWidth = 60; $img.CacheOption = 'OnLoad'; $img.EndInit()
            $w.FindName('Logo').Source = $img
        }
        $w.Add_MouseLeftButtonDown({ try { $w.DragMove() } catch { } })
        $w.FindName('Main').Add_Click({ if ($shared) { $shared.Stop = $true }; $w.Close() })
        if ($shared) {
            $w.FindName('Main').Content = 'Cancel'
            $timer = New-Object Windows.Threading.DispatcherTimer
            $timer.Interval = [TimeSpan]::FromMilliseconds(300)
            $timer.Add_Tick({ if ($shared.Found -or $shared.Stop) { $timer.Stop(); $w.Close() } })
            $timer.Start()
        }
        [void]$w.ShowDialog()
    } catch { }
}

# The search runs in a runspace of its own, so a drive that does not answer cannot freeze the window.
function Start-Search($target, $shared) {
    $rs = [RunspaceFactory]::CreateRunspace(); $rs.Open()
    $rs.SessionStateProxy.SetVariable('shared', $shared)
    $rs.SessionStateProxy.SetVariable('tgt', $target)
    $rs.SessionStateProxy.SetVariable('waitSec', $WaitSec)
    $ps = [PowerShell]::Create(); $ps.Runspace = $rs
    [void]$ps.AddScript({
        function Test-Readable([string]$path) {
            if (-not $path) { return $false }
            try {
                if (-not [IO.File]::Exists($path)) { return $false }
                $fs = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
                try { $b = New-Object byte[] 64; return ($fs.Read($b, 0, 64) -gt 0) } finally { $fs.Dispose() }
            } catch { return $false }
        }
        $t0 = Get-Date
        while (-not $shared.Stop -and ((Get-Date) - $t0).TotalSeconds -lt $waitSec) {
            if (Test-Readable $tgt.app) { $shared.Found = $tgt.app; break }
            foreach ($d in [IO.DriveInfo]::GetDrives()) {
                try { if (-not $d.IsReady) { continue } } catch { continue }
                if (-not $tgt.rel) { break }
                $p = Join-Path $d.Name $tgt.rel
                if (Test-Readable $p) { $shared.Found = $p; break }
            }
            if ($shared.Found) { break }
            Start-Sleep -Milliseconds 700
        }
        $shared.Stop = $true
    })
    $h = $ps.BeginInvoke()
    return @{ ps = $ps; rs = $rs; handle = $h }
}

# --- the round ----------------------------------------------------------------------
try {
    $target = Get-Target
    Write-Log "click: target=[$($target.app)]"
    if ($Diag) {
        $found = Find-App $target
        Write-Output "state dir  : $StateDir"
        Write-Output "target     : $($target.app)"
        Write-Output "readable   : $(if ($found) { $found } else { 'NO' })"
        Write-Output "interpreter: $(if (Test-Path -LiteralPath $PyFile) { (Get-Content -LiteralPath $PyFile -Raw).Trim() } else { 'not cached yet' })"
        Write-Output "app open   : $(Get-RunningApp)"
        return
    }
    if ($WhichPython) {
        $where = if ($App) { $App } else { Find-App $target }
        $py = Find-Python $where
        if ($py) { Write-Output $py }
        return
    }

    $appPath = Find-App $target
    if (-not $appPath) {
        Write-Log ("folder not reachable yet; google drive: " + (Start-GoogleDrive))
        $shared = [hashtable]::Synchronized(@{ Found = $null; Stop = $false })
        $job = Start-Search $target $shared
        Start-Sleep -Milliseconds 1200
        if (-not $shared.Found -and -not $shared.Stop) {
            Show-Box 'ImpactGuide' 'Waiting for Google Drive to mount the project folder. It can take a few minutes after a restart.' $false $shared
        }
        $shared.Stop = $true
        try { [void]$job.ps.EndInvoke($job.handle) } catch { } finally { $job.ps.Dispose(); $job.rs.Close() }
        $appPath = $shared.Found
        if (-not $appPath) { Write-Log 'gave up waiting for the folder'; return }
    }

    $open = Get-RunningApp
    if ($open) {
        Write-Log ("already open (pid $open): " + $(if (Show-Existing $open) { 'brought to the front' } else { 'no window to raise yet' }))
        return
    }

    $py = Find-Python $appPath
    if (-not $py) {
        Write-Log 'no interpreter with slantui, OpenCV and PyQt6-WebEngine'
        Show-Box 'ImpactGuide did not start' 'No Python found with the packages of 04_code/requirements.txt. Install them, or set IMPACTGUIDE_PYTHON.' $true $null
        return
    }
    $pyw = Join-Path (Split-Path -Parent $py) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $pyw)) { $pyw = $py }
    try { if (Test-Path -LiteralPath $ErrFile) { Remove-Item -LiteralPath $ErrFile -Force } } catch { }

    $made = $false
    $ready = New-Object Threading.EventWaitHandle($false, 'ManualReset', 'Local\ImpactGuideUp', [ref]$made)
    [void]$ready.Reset()
    $proc = Start-Process -FilePath $pyw -ArgumentList ('"' + $appPath + '"') -WorkingDirectory $StateDir `
                          -PassThru -RedirectStandardError $ErrFile
    Write-Log "started pid $($proc.Id): $pyw"
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt 120) {
        if ($ready.WaitOne(0)) { Write-Log ("up after {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds); break }
        $proc.Refresh()
        if ($proc.HasExited) {
            $tail = ''
            try { $tail = (Get-Content -LiteralPath $ErrFile -Tail 12) -join "`n" } catch { }
            Write-Log "exited with code $($proc.ExitCode)"
            Show-Box 'ImpactGuide did not start' ("Exit code $($proc.ExitCode).`n" + $tail) $true $null
            break
        }
        Start-Sleep -Milliseconds 150
    }
    $ready.Dispose()
} catch {
    Write-Log "launcher error: $($_.Exception.Message)"
}
