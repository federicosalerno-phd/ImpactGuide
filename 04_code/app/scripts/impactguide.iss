; The Windows installer of ImpactGuide (Inno Setup 6). build_installer.py lays out the stage and runs
;   ISCC.exe /DStage=<stage folder> /DOutDir=<folder> /DOutName=<file name> /DAppVersion=<version> impactguide.iss
; Installed for the user alone, no administrator needed, in %LOCALAPPDATA%\Programs\ImpactGuide; a shortcut on the
; desktop (a task, on by default) and in the Start menu, with the app's icon. What is recorded with the app goes into
; Documents\ImpactGuide (scripts/impactguide_start.pyw), which the uninstaller never touches; the uninstaller removes
; the program, its Python and its caches, and asks before deleting what is left in the program folder (kept when it
; runs silently).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef Stage
  #define Stage "stage\ImpactGuide"
#endif
#ifndef OutDir
  #define OutDir "."
#endif
#ifndef OutName
  #define OutName "ImpactGuide-" + AppVersion + "-setup"
#endif

[Setup]
AppId={{9B6E2F4C-5D1A-4E7B-8C3F-2A6D0E1B7C45}
AppName=ImpactGuide
AppVersion={#AppVersion}
AppVerName=ImpactGuide {#AppVersion}
AppPublisher=Federico Salerno
AppPublisherURL=https://github.com/federicosalerno-phd/ImpactGuide
AppSupportURL=https://github.com/federicosalerno-phd/ImpactGuide
DefaultDirName={localappdata}\Programs\ImpactGuide
DefaultGroupName=ImpactGuide
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
OutputDir={#OutDir}
OutputBaseFilename={#OutName}
SetupIconFile={#Stage}\impactguide.ico
UninstallDisplayIcon={app}\impactguide.ico
UninstallDisplayName=ImpactGuide
LicenseFile={#Stage}\LICENSE.txt
Compression=lzma2
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
AppMutex=ImpactGuideRunning
CloseApplications=yes
SetupLogging=yes

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"

[CustomMessages]
en.Simulation=ImpactGuide simulation
it.Simulation=ImpactGuide simulazione
fr.Simulation=ImpactGuide simulation
es.Simulation=ImpactGuide simulación
de.Simulation=ImpactGuide Simulation
en.DeleteData=Also delete what is left in the program folder (dataset packages, caches)? The data in Documents\ImpactGuide stays.
it.DeleteData=Eliminare anche ciò che resta nella cartella del programma (pacchetti del dataset, cache)? I dati in Documenti\ImpactGuide restano.
fr.DeleteData=Supprimer aussi ce qui reste dans le dossier du programme (paquets du jeu de données, caches) ? Les données de Documents\ImpactGuide restent.
es.DeleteData=¿Eliminar también lo que queda en la carpeta del programa (paquetes del dataset, cachés)? Los datos de Documentos\ImpactGuide se conservan.
de.DeleteData=Auch löschen, was im Programmordner übrig ist (Datensatzpakete, Caches)? Die Daten in Dokumente\ImpactGuide bleiben.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; an older version's program goes first, so that nothing of it stays beside the new one
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\04_code"
Type: filesandordirs; Name: "{app}\scrcpy"
Type: filesandordirs; Name: "{app}\twin"

[Files]
; with 03_data: the protocol and the targets the app copies into Documents\ImpactGuide where they are missing
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\ImpactGuide"; Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\impactguide_start.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\impactguide.ico"; AppUserModelID: "ImpactGuide.Bench"
Name: "{autoprograms}\{cm:Simulation}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\impactguide_start.pyw"" --simulate ""{localappdata}\ImpactGuide\simulation"""; WorkingDir: "{app}"; IconFilename: "{app}\impactguide.ico"; AppUserModelID: "ImpactGuide.Bench"
Name: "{autodesktop}\ImpactGuide"; Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\impactguide_start.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\impactguide.ico"; AppUserModelID: "ImpactGuide.Bench"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-E -s ""{app}\impactguide_start.pyw"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,ImpactGuide}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; what the program made while it ran: its caches and the work it can make again
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\04_code"
Type: filesandordirs; Name: "{app}\02_cad"
Type: filesandordirs; Name: "{app}\03_data"
Type: filesandordirs; Name: "{app}\scrcpy"
Type: filesandordirs; Name: "{app}\twin"
Type: filesandordirs; Name: "{app}\work\live_check"
Type: filesandordirs; Name: "{app}\work\live_models"
Type: filesandordirs; Name: "{app}\work\synth"
Type: filesandordirs; Name: "{app}\work\targets"
Type: filesandordirs; Name: "{app}\work\tests"
Type: filesandordirs; Name: "{localappdata}\ImpactGuide\pycache"
Type: dirifempty; Name: "{app}\work"
Type: dirifempty; Name: "{app}"

[Code]
// What is left in the program folder (the dataset's packages under work, caches) goes only when the operator says so;
// a silent uninstall keeps it. Documents\ImpactGuide, where the trials and calibrations are, is never touched.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and not UninstallSilent then
    if MsgBox(CustomMessage('DeleteData'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    begin
      DelTree(ExpandConstant('{app}'), True, True, True);
    end;
end;
