# TBC — Ternary Behavioral Compilation
## Compilación Conductual Ternaria
### Especificación Técnica Completa — Implementación Post-Training Discreta

> **Versión:** 1.0-técnica — Derivada directa de la propuesta original, preservando la totalidad de ideas.
> **Principio rector:** TBC no cuantiza pesos; compila una función neuronal en pesos ternarios.

---

## 0. Resumen de Corrección de Diseño

Esta especificación corrige una ambigüedad del documento original en los puntos que hablan de "alivianar carga" (puntos 4, 5, 12, 13, 14, 24, 25, 26).

**Corrección formal:** Ningún mecanismo de TBC reduce capacidad funcional, cobertura de verificación o completitud end-to-end. Todas las optimizaciones descritas deben interpretarse bajo el siguiente invariante:

> **Invariante de Memoria Fija:** Una vez iniciado `TBC_COMPILE`, el compilador opera con una asignación de memoria (VRAM/RAM) fija, estable y determinista. Streaming checkpoints, sparse checkpointing, cache jerárquica y evaluación incremental no son atajos que eliminan trabajo, son estrategias de planificación para ejecutar el trabajo completo dentro de un presupuesto de memoria acotado y sin explosión de estado.

El sistema es extremo-a-extremo, sin disminución de capacidades.

---

## 1. Qué Construye Realmente TBC

**Entrada:**
```
M_FP : Modelo en FP16 / BF16 / FP32 con arquitectura A y parámetros W ∈ R^N
```

**Salida:**
```
M_T : Modelo con architecture(M_T) = architecture(M_FP)
      W_T ∈ {-1, 0, +1}^N
      + conjunto de escalas α (por tensor / canal / grupo / bloque)
      + manifiesto de compilación
```

TBC no es:
```
LLaMA FP16 -> aplastar pesos -> LLaMA ternario
```

TBC es:
```
M_FP -> TRACE GENERATOR -> Especificación Conductual C -> Compilador TBC -> M_T
```
M_T nace como producto compilado, no como peso redondeado.

**Propiedad de preservación arquitectónica:**
```
∀ layer l : shape(W_T[l]) = shape(W_FP[l])
∀ op : op_T = op_FP
```

## 2. La Condición Fundamental — Optimización Conductual

Sea:
```
y = F(x; W)
ŷ = F(x; Ŵ) ,  Ŵ_ij ∈ {-1,0,+1}
```

Objetivo de compilación:
```
Ŵ* = argmin_{Ŵ ∈ {-1,0,+1}^N} D( F(x;W), F(x;Ŵ) )
sujeto a D(...) ≤ ε
```

Donde ε ≈ 5% como tolerancia inicial.

D no es distancia de pesos. Es distancia conductual compuesta:
```
D = λ_L * D_logits + λ_H * D_hidden + λ_A * D_activation + λ_B * D_block
con λ_* ≥ 0 , Σλ = 1
```

**Diferencia clave vs distillation:** No se optimiza para enseñar. Se resuelve una restricción de satisfacción: encontrar representación ternaria que satisface observaciones del modelo de referencia.

## 3. El Compilador TBC — Arquitectura de 6 Componentes

```
┌─────────────────────────┐
│     MODELO ORIGINAL     │
│      M_FP (FP16/32)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    TRACE GENERATOR      │ -> Generación de estímulos y captura
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    BEHAVIOR CACHE       │ -> Almacenamiento acotado, fijo en memoria
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  SENSITIVITY ANALYZER   │ -> S_l por capa/bloque/canal
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ TERNARY SEARCH ENGINE   │ -> Búsqueda discreta estructurada
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  EQUIVALENCE CHECKER    │ -> Verificación con tolerancia progresiva
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      TERNARY MODEL      │ -> M_T + Manifest
└─────────────────────────┘
```

### 3.A Trace Generator — Especificación

Captura checkpoints en el grafo de cómputo:
```
x -> embedding -> L1 -> L2 -> ... -> L32 -> logits
                  └────┬─────┘
                       └─> checkpoints {h_l, A_l, z}
```

Interfaz:
```python
Trace = { layer_id: l, hidden: H_l, activation: A_l, logits: z }
GeneratorConfig = { batch_size, seq_len, coverage_target, p_sparse }
```

### 3.B Behavior Cache — Diseño de Memoria Fija

Ver Sección 4.

### 3.C Sensitivity Analyzer
Calcula S_l : Importancia conductual.

### 3.D Ternary Search Engine
Coord. Search + Beam + Branch-and-Bound + Evaluación Incremental.

### 3.E Equivalence Checker
Implementa métrica E compuesta (Sección 16) y tolerancia progresiva (Sección 17).

## 4. Cómo Evitar la Explosión de Memoria — Streaming Checkpoints con Presupuesto Fijo

**Problema:** Modelo 70B, si se guardan todos los hidden states + activaciones + logits = O(T * L * d_model) inmanejable.

**Solución Técnica Implementable:**

TBC NO hace:
```
ejecutar todo -> guardar todo -> procesar después
```

TBC implementa **Streaming Aggregation con Memoria Estable:**

```
for batch in calibration_set:
    h = forward(M_FP, batch)
    summary = Summarize(h)   # estadísticos compactos
    if is_exact_checkpoint(l):
        cache.write_exact(h_l) con política LRU fija
    else:
        cache.write_summary(summary)
    free(h)  # liberación inmediata
```

Definición formal de resumen:
```
A_l^cache = Summary(A_l)
Summary(A_l) = {
  mean, var, norm_l2,
  histograma por canal,
  top-k valores/indices,
  proyección reducida (PCA/random projection),
  muestras aleatorias estratificadas
}
```

Invariante:
```
memoria(TBC) ≤ MEM_BUDGET  ∀ t > t_init
alloc dinámico = 0 después de init
```

Estructura:
```
            RAM/VRAM (presupuesto fijo)
                 │
     ┌───────────┴───────────┐
     │                       │
estadísticas             checkpoints
  compactas                exactos (circular buffer)
     │                       │
     └───────────┬───────────┘
                 ▼
              compiler
```

Esto preserva completitud: el compilador tiene acceso a toda la información conductual necesaria, pero en forma agregada + ventanas exactas rotativas.

## 5. Sparse Behavioral Checkpointing — Formalización del 1/3

Idea original: observar solo 1/3 de activaciones por ejecución.

Formalización correcta (sin pérdida de cobertura):

Sea p ≈ 1/3 la fracción de observación exacta por run.

En cada ejecución k, se muestrea un subconjunto:
```
L_k ⊂ {1..L} , |L_k| = p*L
Run 1: {L2, L7, L13, L19, L27}
Run 2: {L1, L8, L12, L22, L30}
Run 3: {L4, L6, L17, L24, L31}
```

Con política de muestreo sin reemplazo estratificada por profundidad:
```
P(l observado tras K runs) = 1 - (1-p)^K
E[coverage] -> 1 cuando K -> log(1-target)/log(1-p)
```

**Corrección clave:** Esto NO es hacer menos trabajo. Es distribuir la observación exacta en el tiempo para mantener memoria fija, garantizando cobertura total acumulada. El trabajo total de verificación permanece idéntico.

Implementación:
```python
class SparseCheckpointScheduler:
    def __init__(self, L, p=0.33, strategy="stratified_depth"):
        self.L = L; self.p = p
    def sample(self, run_id) -> List[int]:
        # retorna p*L capas con rotación determinista
```

## 6. No Necesitas Usar Prompts Reales — Estímulos Sintéticos como Test Suite

El compilador no necesita conversaciones reales. Necesita **especificación conductual**.

Conjunto de pruebas:
```
C = {x1, x2, ..., xN}  # entradas sintéticas controladas
```

Cada entrada genera traza:
```
T(x) = {h1, h2, ..., hL, logits}
```

Propiedades de C:
- Cobertura de longitudes (32, 128, 512, 2048 tokens)
- Cobertura de entropía (baja, media, alta)
- Cobertura de activación (tokens raros, frecuentes, especiales)
- Determinista y reproducible (seed fijo)

Analogía formal:
```
Programa original M_FP : especificación
Test suite C : casos de prueba unitaria
M_T : implementación que pasa tests con ε
```

## 7. La Primera Cuantización No Necesita Ser Inteligente — Semilla de Búsqueda

Cuantización inicial trivial como punto de partida W^(0):

```
Q(w) = {
  +1 si w > τ
   0 si |w| ≤ τ
  -1 si w < -τ
}
τ = percentile(|W|, 33%) o τ = 0.5 * mean(|W|)
```

Pipeline:
```
W_FP -> Q_simple -> W0 -> TBC_iter1 -> W1 -> TBC_iter2 -> W2 -> ...
```

W0 NO es el resultado. Es solo semilla para búsqueda discreta. Toda la inteligencia está en el compilador, no en Q.

## 8. Cómo Hace la Búsqueda sin Probar 3^N Combinaciones

Espacio total: 3^N imposible.

TBC aplica **Búsqueda Estructurada y Localizada:**

Descomposición jerárquica:
```
Layer
 └─ block (ej: 128x128)
     └─ group (ej: 32 pesos contiguos)
         └─ weight
```

Estrategia: optimizar de pequeño a grande, con independencia local y validación global.

Complejidad:
```
O( L * (B * (G * 3^g * K_beam)) )  << 3^N
donde g = group_size (ej: 4-8), K_beam = 4-16
```

## 9. Coordinate Search — Descenso Discreto

Para cada grupo:
```
grupo actual: [-1, +1, 0, 0, +1]
```

El compilador prueba mutaciones de 1-Hamming distance:
```
[ 0,+1,0,0,+1] -> evaluar D
[+1,+1,0,0,+1] -> evaluar D
[-1, 0,0,0,+1] -> evaluar D
[-1,-1,0,0,+1] -> evaluar D
...
```

Regla de aceptación:
```
if D(W_candidate) < D(W_current) - δ_min:
    accept
```

Sin gradiente. Solo:
```
candidate -> evaluate -> compare -> accept/reject
```

Implementable como kernel GPU paralelizado: evaluación de G*2 candidatos simultáneos.

## 10. Beam Search — Evitar Óptimos Locales

Para no quedar atrapado, conservar K mejores configuraciones.

```
          W0
       /  |  \
      /   |   \
    W1   W2   W3   (K=3)
   /|\   /|\   /|\
  ...   ...   ...
```

Parámetros:
```
K_fast = 4
K_standard = 8
K_max = 16
```

Pruning: mantener solo top-K por D compuesto. Memoria fija: K slots pre-asignados.

## 11. Branch-and-Bound — Poda por Límite

Si una modificación produce:
```
D > D_max (ε actual)
```
Abandonar inmediatamente toda esa rama.

```
candidate
    │
    ├── error 1.2% -> continuar
    ├── error 3.7% -> continuar
    └── error 8.4% -> DESCARTAR + no explorar descendientes
```

Implementación requiere bound inferior rápido (cheap check de Sección 14). Reducción de espacio estimada >90% en capas de baja sensibilidad.

## 12. El Truco Más Importante para Velocidad: No Ejecutar el Modelo Entero — Evaluación con Prefijo Cacheado

Al modificar Layer 17, no recalcular 1..32.

Usar checkpoint guardado h16:
```
checkpoint h16 (desde Behavior Cache)
      ↓
Layer17 candidate A
      ↓
compare con original Layer17 (h17_ref)
```

```
h16_cache -> L17_candidate -> ĥ17 -> D( h17_ref , ĥ17 )
```

Costo pasa de O(L) a O(1) por candidato. Ganancia: ~Lx (ej: 32x en LLaMA-32).

**Nota de implementación:** h16 debe ser exacto para la capa objetivo, no resumen. El Sparse Scheduler (Sec 5) garantiza disponibilidad rotativa.

## 13. Cache Jerárquica — Reutilización de Prefijos y Sufijos

Extensión del punto 12:

```
Input -> [L1-L8] cache -> [L9-L16] cache -> L17 candidate -> [L18-L32] (opcional para full check)
```

Al optimizar L17:
```
L1-L16 reutilizados (cache)
L17 recalculado
L18-L32 solo si pasa cheap check
```

Al optimizar L18:
```
L1-L17 reutilizados (nuevo cache incluye mejor L17)
L18 recalculado
```

Estructura de cache:
```python
class HierarchicalCache:
    prefix_cache: Dict[cut_point, Tensor]  # h_8, h_16, h_24...
    exact_cache: CircularBuffer[Tensor]    # últimos K h_l exactos
```

## 14. Evaluación por Capas Antes de Evaluar Logits — Cascada de Checks

No llegar a logits para cada candidato. Cascada early-exit:

```
candidate
   ↓
cheap check (norma L2)
   ↓ ¿pasa? NO -> descartar
   ↓ SÍ
medium check (cosine)
   ↓ ¿pasa? NO -> descartar
   ↓ SÍ
full check (logits top-k + KL)
```

### Nivel A - Norma Relativa:
```
E_A = ||h - ĥ||_2 / (||h||_2 + δ)
```

### Nivel B - Cosine Distance:
```
E_cos = 1 - cos(h, ĥ)
```

### Nivel C - Logits Top-K:
```
E_topk = ||topk(z) - topk(ẑ)||_2 / ||topk(z)||_2
```

### Nivel D - Distribución Completa:
```
E_KL = D_KL(softmax(z) || softmax(ẑ))
```

Mayoría de candidatos muere en A/B (costo ~0.1% de full forward). Solo ~5-10% llega a D.

## 15. Asignación de Esfuerzo por Sensibilidad

Calcular sensibilidad S_l por capa:

```
S_l = E[ ||∂L/∂h_l|| * ||h_l|| ]  aproximado por perturbación o Fisher diagonal
o empírico: Δ salida al ternarizar solo capa l
```

Mapa conceptual:
```
Layer 1   low
Layer 2   low
Layer 3   medium
...
Layer 19  very high
...
Layer 31  medium
```

Política de asignación:
```
if S_l == low:
   K=4, candidates=16, tolerance=10%
elif medium:
   K=8, candidates=64, tolerance=7%
elif very_high:
   K=16, candidates=256, tolerance=5%, cross-layer refinement habilitado
```

Presupuesto computacional sigue importancia conductual, no número de parámetros. Implementable con prioridad en cola.

## 16. Métrica de Equivalencia — Cuatro Escalas

### Error de activación:
```
E_A = ||A - Â||_2 / (||A||_2 + δ)
```

### Error de hidden state:
```
E_H = ||H - Ĥ||_2 / (||H||_2 + δ)
```

### Error de logits:
```
E_L = ||z - ẑ||_2 / (||z||_2 + δ)
```

### Divergencia de salida:
```
E_KL = D_KL(P || P̂)  donde P = softmax(z/T)
```

Combinada:
```
E = λ_A*E_A + λ_H*E_H + λ_L*E_L + λ_KL*E_KL
```

Aceptación:
```
E < ε_target
```

Configuración recomendada inicial: λ_H=0.3, λ_A=0.2, λ_L=0.2, λ_KL=0.3

## 17. Tolerancia Progresiva — Restricción Progresiva de Búsqueda

El comportamiento final importa más que reproducir exactamente cada activación intermedia.

Schedule de ε por iteración global:
```
Iter 1: ε_layer = 10%
Iter 2: ε_layer = 8%
Iter 3: ε_layer = 6%
Iter 4: ε_layer = 5% (target)
```

No es entrenamiento. Es **constraint tightening**. Permite encontrar rápido solución aproximada y luego endurecer.

Implementación:
```python
for epsilon in [0.10, 0.08, 0.06, 0.05]:
    compile_until(epsilon)
    if global_error > epsilon: backtrack()
```

## 18. Modos de Compilación — Fast, Standard, Max

Misma arquitectura, diferente presupuesto:

### TBC-Fast:
```
calibration traces: 512
sensitivity: empírica rápida
search: greedy + coordinate, K=4
branch-and-bound: leve
tiempo objetivo: minutos
```

### TBC-Standard:
```
traces: 2048
sensitivity: Fisher diag + perturbación
search: beam K=8 + B&B completo
incremental eval: sí
tiempo objetivo: <1h para 7B
```

### TBC-Max:
```
traces: 8192
beam: K=16
cross-layer refinement: sí
backtracking: habilitado
tiempo objetivo: horas, máxima calidad
```

## 19. Comparativa Conceptual con Métodos Actuales

| Método | Qué preserva | Tipo |
|--------|--------------|------|
| RTN | pesos | redondeo |
| GPTQ | error ponderado de salida de capa (2nd order) | PTQ |
| AWQ | comportamiento condicionado por activaciones (canales salientes) | PTQ |
| AQLM | salida de capas/bloques con codebooks | PTQ |
| **TBC** | **equivalencia conductual explícita end-to-end** | **compilación discreta** |

TBC se define como:
> **post-training discrete behavioral compilation**

No es otra cuantización. Es compilación.

## 20. Qué Significa "1.58 Bits" Realmente

Con estados {-1,0,+1}:
```
bits_teoricos = log2(3) ≈ 1.58496
```

Almacenamiento físico requiere empaquetado base-3:
```
5 pesos ternarios -> 3^5 = 243 valores -> cabe en 8 bits (256)
Eficiencia: 5*1.585=7.925 bits usados / 8 bits físicos ≈ 99%
```

Formatos:
- 2-bit con desperdicio (simple pero 0.415 bits overhead)
- Packed base-3 (óptimo teórico)
- Formato I2_S / TL2 usado por BitNet.cpp (ver Sec 39)

## 21. La Escala No Tiene por Qué Desaparecer

Aunque W_T ∈ {-1,0,+1}, se conserva escala continua:
```
W ≈ α * Ŵ
donde α puede ser:
  - por tensor (1 float por matriz)
  - por canal (1 float por fila/columna)
  - por grupo (ej: cada 32-128 pesos)
  - por bloque
```

Núcleo computacional:
```
Y ≈ X * (α * Ŵ) = α * (X * Ŵ_ternario)
```
Parte discreta ternaria, α conserva información de magnitud. Mejora reconstrucción ~20-30% sin romper propiedad 1.58-bit (α es overhead despreciable: 1 float32 por 128 pesos = 0.25 bits extra).

## 22. El Compilador Busca También la Escala — Optimización Conjunta

Buscar (Ŵ, α):
```
Ŵ ∈ {-1,0,+1}^N
α* = argmin_α D( XW , XαŴ )
```

Solución analítica para objetivo L2:
```
α* = ( (XW)^T (XŴ) ) / ( ||XŴ||^2 + δ )
o por mínimos cuadrados: α* = mean(|W|_{W_T≠0}) para grupos
```

Costo casi cero: no requiere búsqueda discreta, se calcula cerrado tras elegir Ŵ. Debe implementarse como paso fused en kernel.

## 23. Formulación Matemática Más Completa

Modelo completo:
```
M(x; W)
```

Compilador:
```
Ŵ = Compile(W, C, ε)
C = {x1, ..., xn}  conjunto de pruebas
```

Restricciones:
```
Ŵ_i ∈ {-1,0,+1} ∀ i
D( M(x_j;W), M(x_j;Ŵ) ) ≤ ε  ∀ x_j ∈ C
D_l ≤ ε_l para checkpoints críticos l ∈ L_crit
```

Objetivo:
```
min_{Ŵ} Σ_j D( M(x_j;W), M(x_j;Ŵ) )
s.t. Ŵ_i ∈ {-1,0,+1}
     D_l ≤ ε_l
```

## 24. Algoritmo TBC Completo — Implementación End-to-End con Memoria Fija (CORREGIDO)

**Invariante corregido:** Este algoritmo es completo, sin reducción de capacidades. Las optimizaciones mencionadas son de planificación, no de recorte.

```python
def TBC_COMPILE(model_fp16: Model, config: TBCConfig) -> TBCModel:

    # 0. Init con presupuesto fijo
    mem_budget = config.vram_budget  # ej: 24GB
    cache = HierarchicalCache(budget=mem_budget, policy="fixed")
    scheduler = SparseCheckpointScheduler(L=model.layers, p=0.33)
    
    # 1. Capturar trazas de referencia (streaming, memoria fija)
    reference_traces = []
    for batch_id, x in enumerate(calibration_set C):
        trace = TraceGenerator(model_fp16, x, scheduler.sample(batch_id))
        cache.ingest(trace.summarize())  # nunca crece más allá de budget
        reference_traces.append(trace.get_summaries()) # solo summaries, no tensores completos
    
    # 2. Calcular sensibilidad S_l
    sensitivity_map = SensitivityAnalyzer(model_fp16, reference_traces).compute()
    # S_l por capa, bloque, canal
    
    # 3. Crear W_ternary inicial (semilla)
    W0 = initial_ternarization(model_fp16.W, tau_strategy="percentile_33")
    W_current = W0
    
    # 4. Búsqueda por bloques con cache jerárquica
    for layer_id in order_by_sensitivity(sensitivity_map, strategy="front_to_back"):
        
        h_in = cache.get_prefix(layer_id)  # reutilizado, no recalculado
        
        # 4b. Generar candidatos locales (coordinate search)
        candidates = generate_candidates(
            W_current[layer_id], 
            group_size=config.group_size, 
            hamming_radius=1,
            num_candidates=sensitivity_to_budget(sensitivity_map[layer_id])
        )
        
        # 4c. Evaluación en cascada con memoria fija
        filtered = []
        for cand in candidates:
            # Incremental eval: ΔY = X ΔW
            delta = cand.W - W_current[layer_id].W
            Y_base = cache.get_base_output(layer_id)
            Y_cand = Y_base + (h_in @ delta)  # solo modificación
            
            # Cheap check
            if norm(Y_cand - Y_ref) / norm(Y_ref) > config.epsilon_cheap:
                continue
            # Medium check
            if cosine_dist(Y_cand, Y_ref) > config.epsilon_medium:
                continue
            filtered.append(cand)
        
        # 4e. Beam search + Branch-and-Bound con K fijo
        beam = BeamSearch(K=config.beam_size).search(
            filtered, 
            reference=cache.get_reference(layer_id),
            metric=E_composed,
            bound=config.epsilon_layer
        )
        
        # 4g. Conservar mejor + actualizar cache
        W_current[layer_id] = beam.best()
        cache.update_prefix(layer_id+1, W_current[layer_id]) # propaga hacia adelante
    
    # 5. Repetición front-to-back ya incluida en loop anterior
    # 6. Validación global full-model
    global_error = EquivalenceChecker(model_fp16, TernaryModel(W_current)).validate(
        C, metric=E_composed
    )
    
    # 7. Backtracking si error > tolerancia
    while global_error > config.epsilon_target and budget_remaining():
        bottleneck_layers = sensitivity_map.top_k_failed(global_error.trace)
        for l in bottleneck_layers:
            W_current[l] = refine_layer(l, larger_beam=True, cache=cache)
        global_error = revalidate()
    
    # 8. Criterio de parada
    # hasta error <= epsilon o presupuesto agotado
    
    # 9. Empaquetado ternario
    packed_weights = pack_ternary(W_current, format=config.packing_format) # base-3 / I2_S
    
    # 10. Exportar
    return TBCModel(
        weights=packed_weights,
        scales=extract_scales(W_current),
        manifest=build_manifest(config, sensitivity_map, global_error)
    )
```

## 25. Cómo Hacerlo Realmente Rápido — Prioridad de Optimizaciones

Orden de impacto (implementar en este orden):

1. **Cached layer evaluation** - Reduce O(L) a O(1) por candidato. Más importante.
2. **Greedy discrete coordinate search** - Barato, paralelizable, fused kernels.
3. **Beam search pequeño K=4-16** - Acota explosión combinatoria con memoria fija.
4. **Branch-and-Bound** - Poda >90% ramas inviables con cheap bounds.
5. **Sensitivity-aware allocation** - Cómputo sigue importancia, no conteo de params.
6. **Early-exit scoring** - 90% candidatos mueren en L2 norm sin llegar a logits.
7. **Parallel candidate evaluation** - Batch de candidatos en una sola GEMM ternaria.
8. **GPU fused kernels** - Evitar overhead de lanzar miles de kernels pequeños; kernel único que hace: load h_in + load ΔW + compute ΔY + compute norm + decision.

Meta: minutos en lugar de horas para 7B en single GPU 24GB.

## 26. Optimización Extremadamente Importante: Reutilizar Diferencias — Evaluación Incremental

Si grupo cambia W -> W + ΔW:
```
X(W+ΔW) = XW + XΔW
```

No recalcular multiplicación completa.

Si ΔW es pequeño y local (ej: 4 pesos cambian en grupo de 128):
```
Y_candidate = Y_base + ΔY
donde ΔY = X * ΔW  (matriz pequeña, dispersa)
```

Transforma:
```
de: [seq_len, d_in] x [d_in, d_out] = O(S * d_in * d_out)
a:  [seq_len, g] x [g, d_out_local] = O(S * g)  donde g << d_in
```

**Implementación:** Kernel sparse-dense incremental. Decisivo para pasar de horas a minutos.

Mantiene completitud: resultado matemáticamente idéntico, solo computado más rápido.

## 27. El Compilador Debe Ser Capaz de Retroceder — Backtracking Neuronal

No solo izquierda a derecha.

```
Layer 17 -> Layer 18 -> Layer 19 -> ... -> Layer 25 -> failure (E > ε)
```

Consulta sensitivity map y vuelve:
```
19 -> 22 -> 25 (re-optimizar subgrafo)
```

Implementación como grafo de dependencias:
```python
if global_error.layer[l] > ε_l:
    affected = sensitivity_map.dependents(l)  # capas que alimentan a l
    for a in sorted(affected, key=S, reverse=True):
        recompile(a, with_larger_budget=True)
```

Es búsqueda con backtracking dirigido por sensibilidad, no lineal.

## 28. Artefactos Finales

No solo modelo. Dos artefactos:

### A. Modelo:
```
model.tbc
- pesos empaquetados (ternary packed)
- escalas α por grupo/canal
- formato: compatible bitnet.cpp (ver Sec 39)
```

### B. Manifest del compilador:
```json
{
  "architecture": "LLaMA-7B",
  "ternary_format": "i2_s_tl2_packed",
  "group_size": 32,
  "scales": {"granularity": "per_group", "dtype": "fp16"},
  "checkpoint_layout": {"p": 0.33, "strategy": "stratified_depth"},
  "error_threshold": {"global": 0.05, "per_layer": {"l19": 0.04}},
  "calibration_signature": "sha256(C)",
  "layer_tolerances": {"L1": 0.10, "L19": 0.05},
  "compiler_version": "tbc-1.0.0",
  "packing_format": "bitnet.cpp-compatible",
  "sensitivity_map": {"L19": "very_high", ...},
  "compile_time": "247s",
  "peak_vram": "21.3GB"
}
```

Permite reproducir y validar compilación.

## 29. Qué Resultados Teóricos Podríamos Esperar

Separar esperable matemáticamente vs demostrable experimentalmente. No afirmar superioridad sin prueba.

Ventaja conceptual:

Cuantización convencional minimiza:
```
||W - Ŵ|| o ||XW - XŴ|| local
```

TBC minimiza directamente:
```
||F(x;W) - F(x;Ŵ)|| global
```

Por tanto puede ocurrir:
```
||W - Ŵ|| grande  pero  ||F_W - F_Ŵ|| pequeño
```

Que es exactamente lo deseado: dos conjuntos de pesos numéricamente diferentes que implementan casi la misma función en región de entrada relevante.

## 30. La Gran Hipótesis de TBC

> **A igualdad de representación ternaria y arquitectura, optimizar directamente la equivalencia funcional del modelo puede conservar más comportamiento del modelo original que minimizar únicamente el error de cuantización de los pesos.**

Comprobable experimentalmente comparando bajo mismo presupuesto:

```
FP16
vs
RTN-1.58
vs
GPTQ-ternary / método ternario equivalente
vs
TBC-1.58
```

manteniendo:
```
misma arquitectura
mismo número de pesos
mismo presupuesto de calibración
misma representación final
```

## 31. Qué Métricas Usaría para Demostrarlo

No solo accuracy:

- Perplexity (WikiText2, C4)
- KL divergence P||P̂
- logit cosine similarity
- hidden-state cosine similarity
- layer reconstruction error E_A, E_H

Downstream:
- MMLU, HellaSwag, ARC, etc.

Eficiencia de compilación:
```
compile time (s)
peak VRAM (GB)
model size (GB)
tokens/s en bitnet.cpp (ver Sec 39)
```

Afirmación completa: **mejor calidad por unidad de compilación.**

## 32. Un Resultado Especialmente Interesante

Tabla hipotética que validaría idea (números ilustrativos, no reales):

```
Método       PPL       VRAM      compile   size
FP16         5.2       100 GB    —         14GB
RTN-1.58     8.9        25 GB    1 min     2.1GB
Método X     7.4        25 GB    8 min     2.1GB
TBC          6.3        25 GB    4 min     2.1GB
```

Interpretación:
```
misma compresión + misma arquitectura + mucho menos error + compilación rápida
```

## 33. Hay un Límite Fundamental — Imposibilidad Representacional

No existe garantía de que cualquier modelo FP16 pueda ser representado con pesos ternarios conservando ε=5% en todos los comportamientos.

Porque:
```
R^N  ->  {-1,0,+1}^N
```
Capacidad representacional disminuye drásticamente.

Habrá modelos/capas/transformaciones donde:
```
min_{Ŵ∈{-1,0,+1}^N} D(F_W, F_Ŵ) > ε
```

Es decir: **no existe solución ternaria suficientemente cercana bajo métrica elegida.**

En ese caso TBC debe informar explícitamente:
```
Compilation failed
minimum observed error = 7.3%
target = 5%
bottleneck = Layer 18, Group 4-7
```

No fingir éxito.

## 34. Mapa de Imposibilidad — Diagnóstico

Cuando falla, generar mapa:

```
Layer 1    ✓  E=2.1%
Layer 2    ✓  E=1.8%
Layer 3    ✓  E=3.2%
...
Layer 17   ✓  E=4.9%
Layer 18   ✗  E=7.3%  <- cuello
Layer 19   ✓  E=4.1%
...
```

Valioso científicamente incluso en fallo: dice dónde ternarización pura no alcanza tolerancia.

## 35. Posible Solución Futura: Bit-Width Adaptativo — Extensión No Pura

Aunque objetivo sea 1.58-bit puro, TBC podría detectar:

```
Layer 17 -> ternary suficiente
Layer 21 -> ternary insuficiente
```

Y opcionalmente permitir:
```
{-2,-1,0,+1,+2} (2 bits) o pequeño % de pesos FP16
```

Pero eso es **extensión de TBC**, no TBC puro.

Versión pura debe mantenerse:
```
W ∈ {-1,0,+1}
```
para comprobar hipótesis central.

Extensión adaptativa sería TBC-A (Adaptive).

## 36. Relación con Trabajo Reciente

Dirección pertinente porque ternarización post-training sigue activa en 2026 (W1.58A4, etc). Métodos previos ya demostraron que información de activaciones y optimización por bloques mejora cuantización extrema (GPTQ, AWQ, AQLM). Transformaciones equivalentes (rotaciones) como SpinQuant preservan outputs FP mientras hacen más favorable cuantización.

Intuición TBC alineada:
> **la geometría de los pesos no necesariamente es la representación óptima de la función que queremos preservar.**

## 37. Arquitectura Completa de TBC — Diagrama Final

```
                         ┌──────────────────────┐
                         │     FP16/BF16 LLM    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   TRACE GENERATOR    │  C = {x_i} sintéticos
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
             logits             hidden             activations
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  BEHAVIOR CACHE      │  Memoria FIJA, streaming
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ SENSITIVITY ANALYZER │  S_l
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ INITIAL TERNARIZATION│  W0 = Q_simple(W)
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ TERNARY SEARCH ENGINE│
                         │  - coordinate search │
                         │  - beam K=4-16       │
                         │  - branch-and-bound  │
                         │  - incremental eval  │
                         │  - cached eval       │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ EQUIVALENCE CHECKER  │  E < ε progresivo
                         └──────────┬───────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                       pass                fail
                          │                   │
                          ▼                   ▼
                     next layer          backtrack
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  GLOBAL VALIDATION   │  D(F, F̂) ≤ ε
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │    TBC 1.58 MODEL    │  model.tbc + manifest
                         └──────────────────────┘
```

## 38. Definición Final de la Técnica

> **Ternary Behavioral Compilation (TBC)** es un método post-training de compilación neuronal que transforma un modelo de precisión completa en otro de idéntica arquitectura y pesos ternarios {−1,0,+1} mediante búsqueda discreta y verificación de equivalencia aproximada. En lugar de minimizar exclusivamente el error entre pesos cuantizados y pesos originales, TBC utiliza ejecuciones del modelo de referencia como especificación conductual y busca configuraciones ternarias que reproduzcan sus activaciones, estados internos y logits dentro de una tolerancia definida. La compilación emplea trazas parciales con memoria fija y estable desde init, checkpoints dispersos (p≈1/3) con cobertura total acumulada, análisis de sensibilidad, búsqueda local discreta, beam search, branch-and-bound, evaluación incremental (XΔW) y reutilización de estados cacheados para limitar tanto el tiempo de compilación como el uso de memoria sin reducir capacidad funcional ni completitud end-to-end.

Frase resumen:

> **"TBC no cuantiza pesos; compila una función neuronal en pesos ternarios con presupuesto de memoria fijo y verificación conductual completa."**

---

## 39. Requisito de Compatibilidad Nativa con bitnet.cpp [NUEVA SECCIÓN OBLIGATORIA]

### 39.1 Objetivo

Todo modelo producido por TBC **DEBE** ser ejecutable de forma nativa en [bitnet.cpp](https://github.com/microsoft/BitNet) sin conversión, sin re-entrenamiento y sin kernels custom adicionales. TBC es el compilador, bitnet.cpp es el runtime de referencia.

### 39.2 Formato de Pesos — Compatibilidad Obligatoria

TBC debe exportar pesos en formato **I2_S / TL2** (Ternary-2-bit) exactamente como lo espera bitnet.cpp:

- Valores almacenados: `{-1, 0, +1}` mapeados a 2-bit: `00=0, 01=+1, 10=-1` (según spec de bitnet.cpp, verificar en `bitnet.cpp/src/`)
- Empaquetado: **TL2 (Ternary Lookup 2)** — 4 pesos ternarios por byte o 5 pesos por byte según kernel `TL2`, implementación debe alinearse con `bitnet.cpp` `pack_i2_s`
- Alineación: pesos empaquetados en bloques de `group_size` compatible: **32, 64 o 128** (recomendado 32 para compatibilidad con BitNet-b1.58-3B)
- Archivo: `model.tbc` debe ser renombrable a `ggml-model-i2_s.gguf` o convertible a GGUF I2_S mediante script `tbc_to_gguf_i2s.py` incluido en repo TBC, sin pérdida.

Estructura de exportación:
```
model.tbc/
  ├─ config.json (arquitectura)
  ├─ weights/
  │   ├─ layer_00_attn_q_i2s.bin (packed TL2)
  │   ├─ layer_00_attn_q_scales_fp16.bin (α por grupo)
  │   └─ ...
  └─ manifest.json
```

### 39.3 Escalas α — Formato bitnet.cpp

bitnet.cpp espera escalas por grupo (o por bloque). TBC debe generar:

```
W_real = α_g * W_ternary ,  W_ternary ∈ {-1,0,1}
α_g = float16 o float32 por grupo de tamaño G
```

Requisitos:
- `α_g` calculado analíticamente como en Sec 22, no aprendido
- Dtype de escala: `FP16` para inferencia (compat con `bitnet.cpp` kernel `bitnet_b1_58`)
- Granularidad: `per_group G=32` por defecto para máxima compatibilidad
- Almacenamiento: intercalado o separado según GGUF spec `tensor.scales`

### 39.4 Kernels y Runtime

TBC debe validar que:

1. **Kernel de inferencia usado es `bitnet.cpp` `TL2` / `I2_S`**: `X * W_T` se ejecuta como `int2 * fp16` con lookup, no como dequantización a FP16 completa.
2. **No hay dependencia de kernels custom no incluidos en bitnet.cpp upstream.** Todo lo que TBC produce corre con:
```bash
./llama-b1-58 -m model-i2_s.gguf -p "Hello"
```
3. **Soporte para `bitnet.cpp` quantized matmul**: `bitnet::matmul_i2s_fp16` o equivalente.
4. **Compatibilidad con `llama.cpp` fork de BitNet**: El modelo GGUF debe cargar en `llama.cpp` con soporte BitNet habilitado.

### 39.5 Pipeline de Exportación Nativa

```python
def export_bitnet_cpp_compatible(tbc_model: TBCModel, output_path: str):
    # 1. Verificar que W ∈ {-1,0,1}
    assert is_ternary(tbc_model.W)
    
    # 2. Re-agrupar a group_size=32 (si diferente)
    regrouped = regroup(tbc_model.W, target_group=32)
    scales = recompute_scales(regrouped, method="analytic_l2")
    
    # 3. Pack a I2_S TL2
    packed = pack_i2_s_tl2(regrouped)  # kernel idéntico a bitnet.cpp/tools/pack
    
    # 4. Escribir GGUF con tipo GGML_TYPE_I2_S
    write_gguf(
        path=output_path,
        tensors=packed,
        scales=scales,
        type=GGML_TYPE_I2_S,
        arch=tbc_model.arch
    )
    
    # 5. Validación de carga
    assert bitnet_cpp_can_load(output_path)
    assert perplexity_diff < 0.01  # entre model.tbc y GGUF cargado
```

### 39.6 Validación Obligatoria

Todo release de TBC debe incluir test:

```
- [ ] model.tbc -> gguf-i2_s convierte sin error
- [ ] ./build/bin/llama-b1-58 carga GGUF y genera tokens
- [ ] tokens/s medido con bitnet.cpp bench coincide con BitNet oficial 3B (~70 tok/s CPU)
- [ ] ppl en WikiText2 con bitnet.cpp runtime = ppl con runtime TBC nativo (Δ < 1%)
- [ ] manifest incluye campo "bitnet_cpp_compatible": true y "packing": "I2_S_TL2"
```

### 39.7 Justificación

La compatibilidad nativa con bitnet.cpp es **requisito arquitectónico no negociable** porque:

- Garantiza aceleración real en CPU (x86 TL2 kernels, ARM NEON) sin GPU
- Permite aprovechar ecosistema existente de BitNet (quant, serve, llama.cpp)
- Demuestra que TBC produce modelos 1.58-bit reales, no simulados en FP16
- Habilita métrica final `tokens/s` en hardware commodity, que es parte de la hipótesis de TBC: mejor calidad por unidad de compilación Y por unidad de inferencia.

> **Invariante final:** Si un modelo TBC no carga y corre nativamente en `bitnet.cpp` main branch, la compilación se considera fallida, independientemente de su error conductual E.

