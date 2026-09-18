@echo off
cd /d C:\Users\gerso\Desktop\TBC
if "%1"=="quant" (
  third_party\BitNet\build\bin\llama-quantize.exe --token-embedding-type f32 %2 %3 I2_S 1 1 <NUL
  echo QUANT_RC=%errorlevel%
) else (
  third_party\BitNet\build\bin\llama-cli.exe -m %1 -p "%~2" -n %3 --threads 4 <NUL
  echo CLI_RC=%errorlevel%
)
