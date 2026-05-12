@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo   图片文字替换工具
echo ========================================
echo.
echo 正在启动...
echo 打开浏览器访问 http://127.0.0.1:7860
echo.
python app.py
pause
