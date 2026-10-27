<#
.SYNOPSIS
  Puts ImpactGuide where it can be clicked: an icon on the desktop and in the
  Start menu, with a launcher on the local disk.

.DESCRIPTION
  Everything it installs lives in %LOCALAPPDATA%\ImpactGuide\launch:
    impactguide_boot.py  the start: a window at once, then the app behind it (the app keeps this copy current);
                     the shortcut runs it with the pythonw of the interpreter found here
    launcher.ps1     copied from scripts\, refreshed when its LAUNCHER-VERSION changes
    ImpactGuide.vbs  what the shortcut runs when no interpreter was found: the launcher, with no console
    target.json      where impactguide_app.py is, also without the drive letter
    icon.ico         the app's icon, drawn by launcher_assets.py
    theme.json       the SlantUI roles of the launcher's small window
    python.txt       the interpreter found to work
  Run it again after moving the project. It is safe to run twice.

.PARAMETER Python
  The interpreter to draw the icon with; without it, the launcher's own search.

.PARAMETER Uninstall
  Remove the shortcuts and the local launcher folder; the app is left alone.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File 04_code\app\scripts\install_shortcut.ps1
#>
param(
    [string]$Python = "",
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

$ScriptDir  = Split-Path -Parent $PSCommandPath
$AppDir     = Split-Path -Parent $ScriptDir
$AppPath    = Join-Path $AppDir 'impactguide_app.py'
$StateDir   = Join-Path $env:LOCALAPPDATA 'ImpactGuide\launch'
$AppId      = 'ImpactGuide.Bench'            # the same id impactguide_app.py gives the window
$LnkName    = 'ImpactGuide.lnk'
$DesktopLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) $LnkName
$StartLnk   = Join-Path $env:APPDATA ('Microsoft\Windows\Start Menu\Programs\' + $LnkName)

function Say([string]$text) { Write-Output $text }

# The AppUserModelID on the shortcut, so a pinned icon stays the app's own and
# not that of wscript.exe, which is what the shortcut starts.
function Set-LnkAppId([string]$Lnk) {
    if (-not ('IgLnk.Shortcut' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace IgLnk {
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey { public Guid fmtid; public uint pid; public PropertyKey(Guid g, uint p) { fmtid = g; pid = p; } }
    [StructLayout(LayoutKind.Sequential)]
    public struct PropVariant { public ushort varType; public ushort r1; public ushort r2; public ushort r3; public IntPtr pointerValue; public IntPtr unused; }
    [ComImport, Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore {
        int GetCount(out uint c); int GetAt(uint i, out PropertyKey k); int GetValue(ref PropertyKey k, out PropVariant v);
        int SetValue(ref PropertyKey k, ref PropVariant v); int Commit();
    }
    [ComImport, Guid("0000010b-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPersistFile {
        int GetClassID(out Guid c); [PreserveSig] int IsDirty();
        int Load([MarshalAs(UnmanagedType.LPWStr)] string f, uint m);
        int Save([MarshalAs(UnmanagedType.LPWStr)] string f, [MarshalAs(UnmanagedType.Bool)] bool r);
        int SaveCompleted([MarshalAs(UnmanagedType.LPWStr)] string f); int GetCurFile([MarshalAs(UnmanagedType.LPWStr)] out string f);
    }
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")] public class ShellLink { }
    public static class Shortcut {
        [DllImport("ole32.dll")] static extern int PropVariantClear(ref PropVariant pv);
        public static void Set(string lnk, string id) {
            object o = new ShellLink();
            try {
                IPersistFile pf = (IPersistFile)o; pf.Load(lnk, 2);
                PropertyKey key = new PropertyKey(new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), 5);
                PropVariant pv = new PropVariant(); pv.varType = 31; pv.pointerValue = Marshal.StringToCoTaskMemUni(id);
                int hr = ((IPropertyStore)o).SetValue(ref key, ref pv); PropVariantClear(ref pv);
                if (hr != 0) Marshal.ThrowExceptionForHR(hr);
                hr = ((IPropertyStore)o).Commit(); if (hr != 0) Marshal.ThrowExceptionForHR(hr);
                hr = pf.Save(lnk, true); if (hr != 0) Marshal.ThrowExceptionForHR(hr);
            } finally { Marshal.FinalReleaseComObject(o); }
        }
    }
}
'@ -ErrorAction Stop
    }
    try { [IgLnk.Shortcut]::Set($Lnk, $AppId); return $true } catch { return $false }
}

if ($Uninstall) {
    foreach ($l in @($DesktopLnk, $StartLnk)) {
        if (Test-Path -LiteralPath $l) { Remove-Item -LiteralPath $l -Force; Say "removed $l" }
    }
    if (Test-Path -LiteralPath $StateDir) { Remove-Item -LiteralPath $StateDir -Recurse -Force; Say "removed $StateDir" }
    return
}

if (-not (Test-Path -LiteralPath $AppPath)) { throw "impactguide_app.py is not where expected ($AppPath)." }
if (-not (Test-Path -LiteralPath $StateDir)) { New-Item -ItemType Directory -Path $StateDir -Force | Out-Null }

# --- the launcher, copied on its version line --------------------------------------
$src = Join-Path $ScriptDir 'launcher.ps1'
$dst = Join-Path $StateDir 'launcher.ps1'
$verSrc = (Select-String -LiteralPath $src -Pattern 'LAUNCHER-VERSION:\s*(\d+)' | Select-Object -First 1).Matches[0].Groups[1].Value
$verDst = ""
if (Test-Path -LiteralPath $dst) {
    try { $verDst = (Select-String -LiteralPath $dst -Pattern 'LAUNCHER-VERSION:\s*(\d+)' | Select-Object -First 1).Matches[0].Groups[1].Value } catch { }
}
if ($verSrc -ne $verDst) { Copy-Item -LiteralPath $src -Destination $dst -Force; Say "launcher : version $verSrc installed" }
else { Say "launcher : already at version $verSrc" }
$boot = Join-Path $StateDir 'impactguide_boot.py'
Copy-Item -LiteralPath (Join-Path $ScriptDir 'impactguide_boot.py') -Destination $boot -Force
Say "start    : $boot"

# --- where the app is ----------------------------------------------------------------
$rel = $AppPath
if ($AppPath -match '^[A-Za-z]:[\\/](.+)$') { $rel = $matches[1] }
$json = @{ app = $AppPath; rel = $rel } | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $StateDir 'target.json'), $json, (New-Object Text.UTF8Encoding($false)))
Say "target   : $AppPath"

# --- the interpreter, the icon and the colours ------------------------------------------
$py = $Python
if (-not $py) {
    try { $py = (& powershell -NoProfile -ExecutionPolicy Bypass -File $dst -WhichPython -App $AppPath 2>$null | Select-Object -Last 1) } catch { $py = "" }
    if ($py) { $py = "$py".Trim() }
}
if ($py -and (Test-Path -LiteralPath $py)) {
    Say "python   : $py"
    # Qt may write warnings on stderr: with the preference at Stop the first one would end the install
    $ErrorActionPreference = 'Continue'
    $out = & $py (Join-Path $ScriptDir 'launcher_assets.py') $StateDir 2>&1
    $ErrorActionPreference = 'Stop'
    foreach ($line in $out) { Say ("           " + $line) }
} else {
    Say "python   : none found with slantui, OpenCV and PyQt6-WebEngine (see 04_code/requirements.txt)"
}

# Windows caches a shortcut's icon by path: the shortcut points at a copy named after its content.
$icon = Join-Path $StateDir 'icon.ico'
$iconRef = $icon
if (Test-Path -LiteralPath $icon) {
    $hash = (Get-FileHash -LiteralPath $icon -Algorithm SHA256).Hash.Substring(0, 8).ToLower()
    $iconRef = Join-Path $StateDir ("icon-" + $hash + ".ico")
    if (-not (Test-Path -LiteralPath $iconRef)) { Copy-Item -LiteralPath $icon -Destination $iconRef -Force }
    Get-ChildItem -LiteralPath $StateDir -Filter 'icon-*.ico' | Where-Object { $_.FullName -ne $iconRef } |
        ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Force } catch { } }
}

# --- what the shortcut runs -------------------------------------------------------------------
# The start itself, with pythonw (no console): PowerShell alone takes a second or two to begin, and wscript with a
# .vbs before it up to another one on a busy computer, which is what kept the window from appearing at the click.
# Without an interpreter, wscript with the .vbs below, which runs the launcher with no console flashing (window
# style 1 for the start: a hidden start, 0, would hide its window too).
$pyw = ""
if ($py -and (Test-Path -LiteralPath $py)) {
    $pyw = Join-Path (Split-Path -Parent $py) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $pyw)) { $pyw = "" }
}
$vbs = Join-Path $StateDir 'ImpactGuide.vbs'
$lines = @(
    "' ImpactGuide: the launch with no console. install_shortcut.ps1 writes this file.",
    'Dim fso, sh, here, pyw, boot',
    'Set fso = CreateObject("Scripting.FileSystemObject")',
    'Set sh = CreateObject("WScript.Shell")',
    'here = fso.GetParentFolderName(WScript.ScriptFullName)',
    ('pyw = "' + $pyw + '"'),
    'boot = here & "\impactguide_boot.py"',
    'If pyw <> "" And fso.FileExists(pyw) And fso.FileExists(boot) Then',
    '    sh.Run """" & pyw & """ """ & boot & """", 1, False',
    'Else',
    '    sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -STA -File """ & here & "\launcher.ps1""", 0, False',
    'End If'
)
[IO.File]::WriteAllText($vbs, ($lines -join "`r`n"), [Text.Encoding]::ASCII)

$bootFile = Join-Path $StateDir 'impactguide_boot.py'
function Write-Shortcut([string]$path) {
    $sc = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
    if ($pyw) {
        $sc.TargetPath = $pyw
        $sc.Arguments = '"' + $bootFile + '"'
    } else {
        $sc.TargetPath = "$env:WINDIR\System32\wscript.exe"
        $sc.Arguments = '//B "' + $vbs + '"'
    }
    $sc.WorkingDirectory = $StateDir
    $sc.IconLocation = $iconRef
    $sc.Description = 'ImpactGuide'
    $sc.Save()
    [void](Set-LnkAppId $path)
}
Write-Shortcut $DesktopLnk
Write-Shortcut $StartLnk
Say "shortcut : $DesktopLnk"
Say "shortcut : $StartLnk"
Say ""
Say "Check it without starting anything:"
Say "  powershell -ExecutionPolicy Bypass -File `"$dst`" -Diag"
