# third_party — bitnet.cpp upstream (no incluido en el repo)

El runtime de referencia **no** se versiona aquí (pesa GB con el build).
Para reproducir los resultados, clónalo y aplica el fix de build:

```bash
git clone --recursive https://github.com/microsoft/BitNet.git third_party/BitNet
cd third_party/BitNet
git submodule update --init --depth 1 --recursive

# Fix TBC (1 línea, solo build, no toca kernels):
# añade ../../../../src/ggml-bitnet-mad.cpp junto a ggml-bitnet-lut.cpp
# en 3rdparty/llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt
patch -p1 < ../bitnet-cmake-fix.patch   # o edítalo a mano (ver el .patch)
```

Compilar (flags oficiales de `setup_env.py`):

```bat
rem Windows (VS + clang MinGW + Ninja) — ver scripts/build_bitnet.bat
```

```bash
# Linux
cmake -B build -G Ninja -DBITNET_X86_TL2=OFF -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release -j
./build/bin/llama-cli --help
```

Sin el fix, el link falla (`undefined symbol: ggml_bitnet_mul_mat`) porque
`ggml-bitnet-lut.cpp` en main solo define ese entry point bajo `ARM_TL1`.
Detalle completo en `README.md §3`.
