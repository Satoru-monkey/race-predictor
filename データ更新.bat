@echo off
chcp 65001 > nul
cd /d %~dp0
echo 新しい結果を取り込んで、モデルを学び直します...
python -m boatrace.update
pause
