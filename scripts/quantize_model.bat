@echo off
echo === Quantisation Qwen 3.5 9B ===
echo.

cd /d "%~dp0\.."

set CONVERT=llama.cpp\convert_hf_to_gguf.py
set QUANTIZE=llama.cpp\build\bin\Release\llama-quantize.exe
set MODEL_DIR=D:\models\qwen3.5-9b
set GGUF_F16=D:\models\qwen3.5-9b-f16.gguf
set GGUF_Q4=D:\models\qwen3.5-9b-q4km.gguf

echo Etape 1 : Conversion safetensors -^> GGUF F16
if exist "%GGUF_F16%" (
    echo   GGUF F16 existe deja, skip.
) else (
    python "%CONVERT%" "%MODEL_DIR%" --outfile "%GGUF_F16%"
    if errorlevel 1 (
        echo [ERREUR] Conversion echouee.
        pause
        exit /b 1
    )
)

echo.
echo Etape 2 : Quantisation F16 -^> Q4_K_M
if exist "%GGUF_Q4%" (
    echo   GGUF Q4_K_M existe deja, skip.
) else (
    "%QUANTIZE%" "%GGUF_F16%" "%GGUF_Q4%" Q4_K_M
    if errorlevel 1 (
        echo [ERREUR] Quantisation echouee.
        pause
        exit /b 1
    )
)

echo.
echo [OK] Modele quantise : %GGUF_Q4%
for %%A in ("%GGUF_Q4%") do echo Taille : %%~zA bytes

echo.
echo Tu peux maintenant supprimer le F16 intermediaire si tu veux :
echo   del "%GGUF_F16%"

pause
