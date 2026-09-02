@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b %errorlevel%
"C:\Users\jkros\miniconda3\envs\pytorch_env\Scripts\cmake.exe" --build build\emc-cuda-ninja --parallel 4
