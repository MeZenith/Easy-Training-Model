@echo off
chcp 65001 >nul
echo Building installer...
"C:\InnoSetup\ISCC.exe" "EasyTraining.iss"
echo.
echo Output: dist\EasyTraining_Setup.exe
pause
