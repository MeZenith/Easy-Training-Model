[Setup]
AppName=Easy Tinking
AppVersion=1.0
AppPublisher=BlueCorner Studio
DefaultDirName={autopf}\EasyTinking
DefaultGroupName=Easy Tinking
OutputDir=D:\python\Easy Training\dist
OutputBaseFilename=EasyTinking_Setup
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
Source: "D:\python\Easy Training\dist\EasyTinking\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Easy Tinking"; Filename: "{app}\EasyTinking.exe"
Name: "{group}\Uninstall Easy Tinking"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Easy Tinking"; Filename: "{app}\EasyTinking.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\EasyTinking.exe"; Description: "Launch Easy Tinking"; Flags: nowait postinstall skipifsilent
