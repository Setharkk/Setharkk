@echo off
echo === Compilation de llama.cpp avec CUDA ===
echo.

set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

cd /d "%~dp0\.."

if not exist llama.cpp (
    echo Clonage de llama.cpp...
    git clone https://github.com/ggml-org/llama.cpp
)

cd llama.cpp

echo.
echo Configuration CMake...
%CMAKE% -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native

echo.
echo Compilation (Release)...
%CMAKE% --build build --config Release -j

echo.
if exist "build\bin\Release\llama-server.exe" (
    echo [OK] llama-server compile avec succes.
    echo [OK] llama-quantize compile avec succes.
) else (
    echo [ERREUR] La compilation a echoue.
)

pause
