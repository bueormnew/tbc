"""Calibración con texto real en distribución (mejora documentada sobre ids aleatorios).

Sec 6 del MD permite estímulos sintéticos; la literatura PTQ y nuestros
experimentos (Standard sobre ids aleatorios empeora PPL de texto) muestran que
la calibración debe seguir la distribución de despliegue. Este módulo provee
un corpus inglés fijo + ventanas deterministas para calibración y evaluación.
"""
from __future__ import annotations
import torch

CORPUS = """
Language models learn patterns from large collections of text. During training they adjust
millions of parameters to predict the next word in a sentence. This simple objective gives
rise to surprisingly fluent prose, summaries, translations, and even fragments of computer code.
Quantization reduces the memory footprint of these models by storing weights with fewer bits.
Ternary quantization goes further, allowing only three values: minus one, zero, and plus one.
The challenge is to preserve the behavior of the original model despite this coarse grid.
Behavioral compilation addresses the problem as discrete search: candidate ternary matrices
are tested against recorded activations of the full precision network.
A beam of the best candidates is kept while worse branches are pruned early.
Caches with fixed memory budgets make the process fit on modest hardware.
Perplexity measures how surprised a model is by new text; lower is better.
Small models trained on tiny datasets still exhibit the same trade-offs between size,
speed, and quality that govern their larger cousins. Calibration data should resemble
the text the model will face in practice, otherwise optimization overfits to noise.
The quick brown fox jumps over the lazy dog near the river bank at dawn.
Engineers measure compile time, peak memory, model size, and tokens per second.
A manifest records every choice so that any compilation can be reproduced exactly.
When a tolerance cannot be met, an impossibility map points at the bottleneck layers.
Research progresses one honest experiment at a time, reporting failures and limits.
Transformers process sequences with attention, feed-forward blocks, and normalization.
Each layer transforms hidden states that flow from embeddings to output logits.
Rotary encodings inject position information without learned position tables in some models,
while older architectures add learned position embeddings to every input token.
Attention heads compare queries against keys and mix values accordingly.
Feed-forward networks expand the hidden size, apply a nonlinearity, and project back.
Residual connections carry the original signal around each sublayer for stability.
Layer normalization keeps activations in a range that trains reliably.
Tokenizers split raw characters into pieces drawn from a fixed vocabulary.
Rare words break into smaller fragments while common words survive as single tokens.
Generation samples one token at a time, feeding each choice back as new context.
Temperature, top-k, and top-p shape how adventurous or conservative sampling becomes.
Evaluation suites probe grammar, facts, reasoning, and resistance to confusion.
A tiny model of one million parameters fits easily but knows very little.
A model of one hundred twenty four million parameters writes short coherent passages.
Scaling laws describe how loss falls as parameters, data, and compute grow together.
No single metric tells the whole story; quality, cost, and speed trade against each other.
Open weights let anyone inspect, quantize, and run a model on local machines.
Reproducibility demands fixed seeds, logged versions, and checksums of artifacts.

Distillation trains a small student to imitate a large teacher on real examples.
Pruning removes weights or neurons that contribute little to the final output.
Sparse models skip multiplications by zero and save both time and energy.
Hardware accelerators exploit low precision arithmetic for faster inference.
Central processing units handle small batches with low latency and no special setup.
Graphics processors shine on large parallel workloads with thousands of threads.
Memory bandwidth often limits generation speed more than raw arithmetic throughput.
Batching many requests together raises throughput at the cost of latency.
Caching keys and values avoids recomputing attention over past tokens.
Quantized key-value caches extend the context that fits in memory.
Long contexts stress position encodings and attention approximations alike.
Sliding windows, sparse patterns, and linear variants tame quadratic attention cost.
Retrieval augmentation grounds answers in external documents supplied at query time.
Tool use lets models call calculators, search engines, and code interpreters.
Agents chain multiple model calls with observations and actions in a loop.
Evaluation harnesses compare outputs against references with string and model based scores.
Human preferences guide fine tuning through rankings and pairwise comparisons.
Safety tuning reduces harmful outputs while preserving helpful capabilities.
Red teams probe for failures before wide deployment of capable systems.
Documentation should state training data, limits, and intended uses clearly.
Versioned releases make regressions visible and rollbacks possible.
Benchmarks saturate over time, so harder suites replace them again and again.
Efficiency tracks report energy per token alongside raw speed numbers.
Edge devices run small models offline with strict power and memory budgets.
Browsers now execute compact models locally with dedicated acceleration APIs.
Phones balance model quality against battery drain and thermal limits.
Compilers fuse operators, tile loops, and prefetch data for each target chip.
Profilers reveal whether a workload is bound by memory, compute, or overhead.
Autotuning searches block sizes and layouts empirically for the machine at hand.
Mixed precision keeps sensitive accumulations wide while inputs stay narrow.
Stochastic rounding removes bias when casting gradients to short formats.
Loss scaling preserves tiny gradient values in sixteen bit training runs.
Gradient checkpointing trades recomputation for a smaller memory footprint.
Optimizers with adaptive step sizes converge faster on ravines and plateaus.
Learning rate schedules warm up, decay, and sometimes restart during training.
Weight decay, dropout, and augmentation combat memorization of the training set.
Early stopping picks the checkpoint that generalizes instead of the last one.
Cross validation estimates performance on unseen slices of available data.
Ablation studies remove one component at a time to measure its contribution.
Error analysis on the worst cases guides the next round of improvements.
Data cleaning, deduplication, and filtering shape behavior as much as architecture.
Licenses and consent constrain which texts may be used for training at all.
Watermarking and provenance records help trace generated content back to models.
Open evaluations invite scrutiny that closed testing cannot provide alone.
Replication of published numbers remains the gold standard of credible reporting.
Negative results deserve publication because they prune dead ends for everyone.
Simple baselines often beat elaborate methods when tuned with equal care.
Random seeds change outcomes, so serious claims average over several runs.
Confidence intervals communicate uncertainty better than single point estimates.
Statistical tests guard against mistaking noise for genuine progress.
Every shortcut in measurement eventually surfaces as a surprise in production.
Careful engineering compounds: small reliable gains stack into large advantages.
"""


def text_windows(tok, n_seq: int, seq_len: int, offset_seq: int = 0, stride: int | None = None) -> torch.Tensor:
    """Ventanas deterministas [n_seq, seq_len] del corpus (offset/stride configurables)."""
    ids = tok(CORPUS, return_tensors="pt")["input_ids"][0].tolist()
    if stride is None:
        stride = seq_len // 2
    wins = []
    i = offset_seq
    guard = 0
    while len(wins) < n_seq and guard < n_seq * 4 + 1000:
        guard += 1
        chunk = ids[i:i + seq_len]
        if len(chunk) < seq_len:
            i = 0
            continue
        wins.append(chunk)
        i += stride
    if len(wins) < n_seq:
        raise RuntimeError("corpus insuficiente para calibración pedida")
    return torch.tensor(wins[:n_seq], dtype=torch.long)
