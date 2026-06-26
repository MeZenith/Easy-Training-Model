@echo off
chcp 65001 >nul
echo 正在压缩 EasyTinking...
"C:\Program Files\7-Zip\7z.exe" a -t7z "dist\EasyTinking.7z" "dist\EasyTinking\*" -mx=5 -m0=lzma2 -ms=on
echo.
echo 完成！dist\EasyTinking.7z
pause
