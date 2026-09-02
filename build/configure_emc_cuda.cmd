@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b %errorlevel%
"C:\Users\jkros\miniconda3\envs\pytorch_env\Scripts\cmake.exe" -S cpp -B build\emc-cuda-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" -DCMAKE_PREFIX_PATH=C:\Users\jkros\miniconda3\envs\pytorch_env\Lib\site-packages\torch\share\cmake
