@echo off
chcp 65001 > nul
cd /d %~dp0
echo 競馬の過去データを取得して学び直します。初回は数時間かかります（途中で止めても、次回は続きから取得します）。
python -m keiba.update
pause
