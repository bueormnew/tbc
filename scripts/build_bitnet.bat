@echo off
cd /d C:\Users\gerso\Desktop\TBC\third_party\BitNet
echo CWD=%CD%
if exist build rmdir /s /q build
echo CLEAN_RC=%errorlevel%
cmake -B build -G Ninja -DBITNET_X86_TL2=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00" -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00" -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON
echo CONFIGURE_RC=%errorlevel%
cmake --build build --config Release -j 18
echo BUILD_RC=%errorlevel%
echo BUILD_DONE_MARKER
