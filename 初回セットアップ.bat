@echo off
chcp 65001 > nul
cd /d %~dp0
echo 必要なライブラリを入れます（数分かかります）...
python -m pip install -r requirements.txt
echo.
echo 終わりました。「起動.bat」をダブルクリックしてください。
pause
