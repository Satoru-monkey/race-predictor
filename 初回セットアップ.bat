@echo off
chcp 65001 > nul
cd /d %~dp0
echo 必要なライブラリを入れます...
python -m pip install -r requirements.txt
echo.
echo 過去データの取得と学習を始めます。初回は1〜2時間ほどかかります。
python -m boatrace.update
echo.
echo 終わりました。「起動.bat」をダブルクリックしてください。
pause
