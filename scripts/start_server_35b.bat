@echo off
echo === Setharkk - Demarrage llama-server (Qwen 3.5 35B-A3B) ===
echo Modele : D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf
echo Config : 18 couches GPU + reste en RAM, contexte 32K
echo Port   : 8080
echo.

set LLAMA_SERVER=llama-bin\llama-server.exe

if not exist "%LLAMA_SERVER%" (
    echo [ERREUR] llama-server non trouve dans llama-bin\
    pause
    exit /b 1
)

if not exist "D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf" (
    echo [ERREUR] Modele non trouve : D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf
    pause
    exit /b 1
)

"%LLAMA_SERVER%" ^
    -m D:\models\Qwen3.5-35B-A3B-Q4_K_M.gguf ^
    --host 127.0.0.1 ^
    --port 8080 ^
    -ngl 18 ^
    -c 32768 ^
    --jinja ^
    --temp 0.7 ^
    --top-k 20 ^
    --top-p 0.8 ^
    --min-p 0.0 ^
    --cache-type-k q8_0 ^
    --cache-type-v q8_0

pause
