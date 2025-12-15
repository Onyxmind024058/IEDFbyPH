[Setup]
AppName=IEDFbyPH
AppVersion=1.0.0
AppPublisher=Universität Basel/ Paul Hiret
AppPublisherURL=mailto:paul.hiret@unibas.ch
DefaultDirName={pf}\IEDFbyPH
DefaultGroupName=IEDFbyPH
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=IEDFbyPH_setup
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "dist\IEDFbyPH\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\IEDFbyPH"; Filename: "{app}\IEDFbyPH.exe"
Name: "{commondesktop}\IEDFbyPH"; Filename: "{app}\IEDFbyPH.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\IEDFbyPH.exe"; Description: "Launch RFEA Analyzer"; Flags: nowait postinstall skipifsilent
