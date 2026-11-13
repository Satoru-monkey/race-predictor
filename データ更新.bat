@echo off
chcp 65001 > nul
cd /d %~dp0
echo 競艇の新しい結果を取り込んで、モデルを学び直します。
echo 初回は過去データ（約15か月分）の取得に1〜2時間かかります。
python -m boatrace.update
pause
