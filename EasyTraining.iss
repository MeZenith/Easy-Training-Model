[Setup]
AppName=Easy Training
AppVersion=1.0
AppPublisher=BlueCorner Studio
DefaultDirName={autopf}\EasyTraining
DefaultGroupName=Easy Training
OutputDir=D:\python\Easy Training\dist
OutputBaseFilename=EasyTraining_Setup
SetupIconFile=D:\python\Easy Training\res\icon.ico
WizardImageFile=D:\python\Easy Training\res\installer_banner.bmp
WizardSmallImageFile=D:\python\Easy Training\res\installer_banner.bmp
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableWelcomePage=no
PrivilegesRequired=none

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional:"

[Files]
Source: "D:\python\Easy Training\dist\EasyTraining\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Easy Training"; Filename: "{app}\EasyTraining.exe"
Name: "{group}\Uninstall Easy Training"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Easy Training"; Filename: "{app}\EasyTraining.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\EasyTraining.exe"; Description: "Launch Easy Training"; Flags: nowait postinstall skipifsilent
