# PROMPT MAESTRO PARA OPENCODE — IMPLEMENTACIÓN COMPLETA TBC (Ternary Behavioral Compilation)

Eres un agente de implementación de sistemas de ML de bajo nivel. Tu tarea es implementar **de cero y de forma completa, funcional y real** el sistema descrito en la especificación técnica:

**ARCHIVO DE ESPECIFICACIÓN OBLIGATORIO:**
`/mnt/data/TBC_Ternary_Behavioral_Compilation_Tecnico.md`

Debes leer ese archivo completo antes de escribir una sola línea de código. Ese archivo es la verdad absoluta del sistema. No puedes eliminar, simplificar ni recortar ideas. Debes corregir la interpretación de "alivianar carga" como está definido en la Sección 0 del MD: memoria fija y estable desde init, pero ejecución completa end-to-end.

---

## OBJETIVO FINAL

Implementar el compilador TBC completo que transforma:
`M_FP (FP16) -> M_T con W_T ∈ {-1,0,+1}` mediante búsqueda discreta y verificación conductual, no cuantización por redondeo.

Y demostrar que funciona en una prueba real con un modelo real.

---

## REQUISITOS NO NEGOCIABLES

### 1. IMPLEMENTACIÓN COMPLETA PASO A PASO — NADA SIMULADO

Debes implementar TODOS estos módulos tal cual están en la Sección 3 y 24 del MD:

1.  **TraceGenerator:** Ejecuta M_FP, captura trazas {h_l, A_l, logits} con SparseCheckpointScheduler p=0.33, muestreo estratificado.
2.  **BehaviorCache (MEMORIA FIJA):** Implementa HierarchicalCache con presupuesto fijo. Circular buffer para checkpoints exactos + summaries (mean, var, norm, histograma, top-k, random projection). Invariante: `memoria(TBC) ≤ MEM_BUDGET ∀ t > t_init`. Cero alloc dinámico después de init. Debes implementar monitoreo de memoria y fallar si se excede.
3.  **SensitivityAnalyzer:** Calcula S_l por capa/bloque/canal mediante perturbación y norma. Genera sensitivity_map.
4.  **TernarySearchEngine:** 
    - Coordinate Search discreto (Hamming distance 1)
    - Beam Search con K=4-16 fijo (slots pre-asignados)
    - Branch-and-Bound con bound inferior rápido
    - Evaluación Incremental: `Y_candidate = Y_base + X*ΔW` (kernel sparse-dense)
    - Cached Layer Evaluation: reutiliza `h_{l-1}` cacheado, no forward completo
5.  **EquivalenceChecker:** Implementa cascada de checks (Sección 14): Nivel A (norma L2), B (coseno), C (top-k logits), D (KL). Métrica compuesta `E = λ_A*E_A + λ_H*E_H + λ_L*E_L + λ_KL*E_KL`
6.  **Tolerancia Progresiva y Backtracking:** Schedule ε = [10%, 8%, 6%, 5%]. Si falla, backtracking dirigido por sensitivity_map.
7.  **Escalas α:** Búsqueda conjunta (Ŵ, α) con α* analítico L2.

**PROHIBIDO:** Usar GPTQ, AWQ, bitsandbytes, auto-gptq como núcleo de TBC. Puedes usarlos SOLO para baseline comparativo RTN, pero el compilador TBC debe ser tu código de búsqueda discreta.

### 2. COMPATIBILIDAD CON ARQUITECTURAS REALES

Desde el día 1, el código debe funcionar con al menos:

- **GPT-2 (GPT2LMHeadModel)**
- **LLaMA (LlamaForCausalLM)** — arquitectura principal
- **Qwen2 / Qwen3 (Qwen2ForCausalLM)**

Usa `transformers` AutoModel. El compilador debe detectar arquitectura y adaptar `group_size`, `layer_names` (attn.q_proj, k_proj, v_proj, o_proj, mlp.gate_proj, up_proj, down_proj).

### 3. COMPATIBILIDAD REAL Y NATIVA CON bitnet.cpp — OBLIGATORIO

Esto no es opcional. Implementa exportación 100% compatible con bitnet.cpp upstream.

Debes:
- Clonar `https://github.com/microsoft/BitNet` y compilar `llama-b1-58`
- Implementar `pack_i2_s_tl2()`: mapear {-1,0,1} -> I2_S (00=0, 01=+1, 10=-1), empaquetar 4 pesos por byte (TL2 kernel). Usa exactamente el mismo layout que `bitnet.cpp/src/`
- Soportar `group_size = 32` por defecto (obligatorio para compatibilidad)
- Exportar escalas α como FP16 por grupo
- Implementar `export_bitnet_cpp_compatible()` que escribe GGUF con tipo `GGML_TYPE_I2_S` (usa `gguf` python lib o tu writer). El GGUF debe cargar en `llama.cpp` con soporte BitNet.
- Función `bitnet_cpp_can_load()` que realmente ejecuta el binario y genera tokens. Nada de "simulado".

Estructura de salida:
```
./tbc_output/
  model.tbc/
  model-i2_s.gguf
  manifest.json con "bitnet_cpp_compatible": true
```

### 4. PRUEBAS SINTÉTICAS REALES

Antes de la prueba con modelo real, debes pasar:

- Test 1: Ternarización trivial + α analítico en tensor aleatorio [1024,1024] -> error L2 < 20%
- Test 2: Coordinate Search en capa LLaMA 1 capa -> demuestra reducción de E
- Test 3: Cached evaluation vs full forward -> salida idéntica (diff < 1e-5)
- Test 4: Pack/Unpack I2_S TL2 roundtrip -> 100% idéntico
- Test 5: BehaviorCache con presupuesto 1GB no crece tras init

Todos ejecutados realmente, con logs.

### 5. PRUEBA REAL DEL SISTEMA — LÍMITE ESTRICTO 6GB RAM

**ESTA ES LA PRUEBA FINAL OBLIGATORIA. NO LA PUEDES OMITIR NI SIMULAR.**

**Modelo:** `DedeProGames/NanoDex-1M` (LlamaForCausalLM, ~1M params, ideal para 6GB)

Pasos exactos:

1.  **Descarga real:** `from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained("DedeProGames/NanoDex-1M")`
2.  **Configura TBC con límite 6GB:**
    ```python
    TBCConfig(
      vram_budget="6GB",  # LÍMITE DURO, debes monitorear con psutil/torch.cuda.max_memory_allocated()
      ram_budget="6GB",
      group_size=32,
      beam_size=4,  # K=4 para Fast, luego K=8 para Standard si hay tiempo
      epsilon_target=0.05,
      calibration_samples=512,  # para 1M model, 512 es suficiente
      packing_format="I2_S_TL2"
    )
    ```
3.  **Compilación:** Ejecuta `TBC_COMPILE` completo. Loguea: tiempo por capa, S_l, E por capa, pico de RAM/VRAM. Si pico > 6GB, FAIL y debes optimizar cache (no aumentar límite).
4.  **Comparación de Perplejidad REAL:** Evalúa en WikiText2 (primeras 1000 secuencias) o en 200 samples sintéticos si WikiText2 no cabe:
    - PPL_FP16
    - PPL_RTN-1.58 (baseline simple)
    - PPL_TBC-1.58
    Reporta: `PPL_FP16=..., PPL_RTN=..., PPL_TBC=..., Δ_TBC_vs_FP16=...`
5.  **Rendimiento:**
    - Tamaño modelo: `model_fp16.safetensors` vs `model-i2_s.gguf` (GB)
    - Pico VRAM/RAM durante compile
    - Tiempo de compilación total
    - Tokens/s en bitnet.cpp: `./llama-b1-58 -m model-i2_s.gguf -p "Hello, I am" -n 50` mide tok/s
6.  **Validación bitnet.cpp:** El GGUF generado DEBE cargar y generar texto coherente en bitnet.cpp. Captura salida del binario como prueba.
7.  **Genera reporte final `TBC_REAL_TEST_REPORT.md` con tabla:**
    ```
    | Métrica | FP16 | RTN-1.58 | TBC-1.58 |
    | PPL | ... | ... | ... |
    | Size | ... | ... | ... |
    | Compile Time | - | - | ... |
    | Peak RAM | - | - | 5.8GB (<6GB) |
    | Tokens/s bitnet.cpp | - | - | ... |
    | bitnet.cpp load | - | - | OK/FAIL |
    ```

### 6. EXIGENCIAS DE CALIDAD

- Código en Python, PyTorch, HuggingFace Transformers. Estructura limpia: `tbc/compiler.py`, `tbc/cache.py`, `tbc/search.py`, `tbc/pack.py`, `tbc/export.py`
- Nada de TODOs ni stubs. Todo funcional.
- Logs detallados en cada fase.
- Manejo de errores: si `min_D > ε`, reportar `Compilation failed` con mapa de imposibilidad (Sección 34 del MD), no fingir éxito.
- Debes implementar `tbc_to_gguf_i2s.py`
- El agente debe trabajar paso a paso hasta que rinda realmente como se quiere. Si algo falla, debuggea y corrige, no avances a la siguiente fase.

---

## ENTREGABLES FINALES OBLIGATORIOS

1.  Repo con código TBC completo
2.  `TBC_REAL_TEST_REPORT.md` con métricas reales de NanoDex-1M
3.  `model-i2_s.gguf` generado que carga en bitnet.cpp (incluye log de ejecución de bitnet.cpp)
4.  `manifest.json` con campos de Sección 28 del MD
5.  Logs de pruebas sintéticas

**CRITERIO DE ÉXITO:** Si no hay `model-i2_s.gguf` que cargue en bitnet.cpp + reporte de PPL real + pico RAM <6GB, la tarea se considera INCOMPLETA. No aceptes atajos.

Empieza leyendo el MD y luego implementa en orden: Cache -> TraceGenerator -> SearchEngine -> Pack -> Export -> Test Sintético -> Test Real NanoDex-1M.

¡Implementa todo completo, muy funcional y sin desbordar los 6GB!
