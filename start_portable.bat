@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0maimai_const_predictor_portable.exe" (
    "%~dp0maimai_const_predictor_portable.exe"
) else (
    echo 未找到 maimai_const_predictor_portable.exe
    echo 请先运行 build_portable.ps1 进行打包。
    pause
)
endlocal
