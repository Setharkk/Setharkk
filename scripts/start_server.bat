@echo off
echo === Setharkk - Demarrage llama-server ===
echo Modele : D:\models\qwen3.5-9b-q4km.gguf
echo Port   : 8080
echo.

set LLAMA_SERVER=llama-bin\llama-server.exe

if not exist "%LLAMA_SERVER%" (
    echo [ERREUR] llama-server non trouve dans llama-bin\
    pause
    exit /b 1
)

"%LLAMA_SERVER%" ^
    -m D:\models\qwen3.5-9b-q4km.gguf ^
    --host 127.0.0.1 ^
    --port 8080 ^
    -ngl 99 ^
    -c 131072 ^
    --jinja ^
    --temp 0.7 ^
    --top-k 20 ^
    --top-p 0.8 ^
    --min-p 0.0

pause
