# Low-VRAM Techniques for Generative AI Inference — Fitting a 22B Video Model into 32 GB

This document explains the VRAM-reduction techniques used to fit LTX-2.5 (a 22B DiT) — in its fully resident, real-time serving configuration — from a 48 GB-class GPU down to a 32 GB-class GPU, written so the ideas transfer to other models and inference servers. The speed-side companion is [Design Notes on Making Generative AI Inference Fast](optimization-techniques.en.md); LTX-2.5-specific history and measurements are in the [full acceleration report](acceleration-report-20260910.md) (Japanese).

The end result first (all measured, with free VRAM capped at 31 GB — a headless RTX 5090 equivalent):

| Stage | Resident | Outcome |
|---|---|---|
| Starting point (all-resident `nvfp4-fast`) | ~32.5 GB | **OOMs during weight loading** at 31 GB free |
| Model offload (for comparison) | 22.2 GB peak | Runs, but ~15 s per 5-second chunk = 3x real time — **not viable for realtime** |
| Techniques 1–2 applied | 28.8 GB | Narration completes; **conversation OOMs** (multi-shape) |
| + Technique 4 (fragmentation reclaim) | 28.8 GB | Narration completes; conversation OOMs **when CUDA Graph is enabled** |
| + Technique 3 (TE layer streaming) | **23.6 GB** | **Conversation with CUDA Graph enabled, 27.1 GB peak**, all chunks succeed |

The key observation: model offload — the "general solution" — is unusable for realtime because the weights cross PCIe on every job, making generation 3x slower. There are VRAM reductions that sacrifice speed and VRAM reductions that preserve it; this document covers only the latter.

---

## 1. First, dissect VRAM

"Out of VRAM" is not a monolithic problem. PyTorch's OOM message actually tells you the whole breakdown. From a message we actually hit:

```
Tried to allocate 364.00 MiB. GPU 0 has a total capacity of 47.26 GiB of which
215.19 MiB is free. ... this process has 30.87 GiB memory in use. Of the
allocated memory 29.06 GiB is allocated by PyTorch, with 910.24 MiB allocated
in private pools (e.g., CUDA Graphs), and 1.33 GiB is reserved by PyTorch
but unallocated.
```

Five distinct categories:

1. **Weights (resident)** — model parameters. The biggest block and the main battlefield.
2. **Activations / workspaces (transient)** — tensors alive only during forward, plus cuBLAS-style scratch space. A failure like "Tried to allocate 364 MiB" is usually a workspace allocation for a new shape.
3. **CUDA Graph private pools** — dedicated memory held by captured graphs (910 MB above). **Grows monotonically with the number of captured shapes and is never released.**
4. **Reserved but unallocated** — memory PyTorch reserved but cannot use (1.33 GB above): allocator fragmentation. Near the limit, this is what kills you.
5. **Outside PyTorch** — CUDA context, cuDNN/cuBLAS handles, etc. (30.87 − 29.06 ≈ 1.8 GB above).

Each category has a different remedy: for (1), don't load / shrink / stream; for (3), control captures; for (4), allocator settings. Conflate them and you end up "just quantizing" — trimming where it doesn't help while missing where it does.

---

## 2. Technique 1: Don't load what you don't use (−1.2 GB)

The simplest technique and the most commonly overlooked one.

The LTX-2.5 server unconditionally loaded the latent upsampler (950 MB) and the temporal upsampler (250 MB) onto the GPU at startup. But realtime requests always carry `upscale: false` — **1.2 GB of modules sat resident without ever being forwarded**.

Three steps:

1. Gate the load behind a config flag (`LTX25_LOAD_UPSAMPLERS=0`)
2. When the feature is requested while unloaded, **fail with a clear error** (never break silently)
3. Check for other code paths that depend on it (here, the t2i quality pipeline also used the upsampler and needed its own guard)

Finding "things you never actually use" is trivial: grep each component's call sites and cross-reference against request parameters. Five minutes of investigation for 1.2 GB — the best cost/benefit ratio in this document.

---

## 3. Technique 2: Attack quantization's blind spots (−1.9 GB resident + −0.5 GB transient)

"The text encoder is already bnb 4-bit quantized, so it's light" — that was the false assumption. Summing the safetensors header tensor by tensor:

| Part | Size | Precision |
|---|---|---|
| Language model body (Linear layers) | 5.71 GiB | NF4 (quantized) |
| **Embedding table** [262144×3840] | **1.88 GiB** | **bf16 (not quantized)** |
| Vision tower | 0.04 GiB | — |

**bitsandbytes 4-bit quantization only targets Linear layers**; `nn.Embedding` passes through untouched. A 262k-vocabulary × 3840-hidden embedding table sat on the GPU in bf16.

### 3.1 Embedding tables can live on the CPU

An embedding is not a matrix multiplication — it is a **table lookup**. A 1024-token prompt touches 1024 rows (~7.5 MB) of the 1.88 GB table. So replace the embedding module's forward:

```python
orig_forward = embed.forward          # class forward (gather + scale multiply)
def bridged_forward(input_ids):
    return orig_forward(input_ids.to("cpu")).to(input_ids.device)
embed.to("cpu")
embed.forward = bridged_forward
```

Send the indices to the CPU, gather there, return only the result to the GPU. 7.5 MB per call; measured speed impact: zero. **The output is bit-identical to GPU execution** (gather is a value copy, and the bf16 scale multiply rounds identically on CPU and CUDA — we verified this by measurement, not assumption).

### 3.2 The trap: the "cleaner design" is the one that breaks

Our first design was "compute `inputs_embeds` ourselves and pass it to the model" — transformers officially supports it, so it looks like the principled choice. But reading the source: on the path without `input_ids`, multimodal-token detection switches to **comparing against embedding vectors**, which calls `get_input_embeddings()(index tensor on cuda)`. The moment the embedding lives on the CPU, that comparison dies with a device mismatch.

The module-forward replacement approach keeps **every internal call site working unmodified**, including that hidden one. Lesson: when you relocate a weight, enumerate every reference path to it in the source. "The main path works" does not mean "all paths work."

### 3.3 A freebie: the all-token logits

When this model is used as a text encoder, only the hidden states are read. Yet `ForConditionalGeneration.forward` always computes `lm_head(hidden_states)` with `logits_to_keep=0` (= all tokens): a **~0.5 GB transient tensor (1024 tokens × 262144 vocab, bf16) created and discarded on every call**.

We replaced `text_encoder.forward` with a thin wrapper that calls the inner model with identical arguments and returns its output — never touching `lm_head`. Note that `lm_head.weight` is tied to the embedding table (the same tensor), so there is no additional weight saving — and tied weights also mean "moving one moves the other," which you need to keep in mind.

---

## 4. Technique 3: Windowed layer streaming (−5.0 GB, the main event)

The remaining heavyweight is the language model body (5.7 GB in NF4). Every layer is used on every generation, so "don't load it" is impossible. But **not all layers are needed at the same time** — they execute in order. So: keep the weights resident in pinned host memory and stream them to the GPU layer by layer, only during encoding.

### 4.1 Do the arithmetic before writing code

Whether streaming can preserve speed is computable in advance:

- Compute time per layer: 0.21 s total encode ÷ 48 layers ≈ **4.4 ms**
- Transfer time per layer: 5.7 GB ÷ 48 ≈ 120 MB; pinned transfers at ~50 GB/s ≈ **2.4 ms**

**Transfer < compute**, so prefetching one layer ahead hides the transfers completely behind computation. If the inequality goes the other way, streaming is guaranteed slower and this technique does not apply. The answer depends on your model and your PCIe generation — measure both sides on your own hardware.

### 4.2 Why the off-the-shelf mechanism was too slow

diffusers ships `apply_group_offloading` for exactly this purpose, and we tried it first (measuring the existing tool before building your own is always right). Results:

| Configuration | Encode time | Problem |
|---|---|---|
| All-resident (baseline) | 0.213 s | — |
| diffusers, use_stream=True | 1.25 s | Streams **force one layer per group**: 48 groups × ~22 ms fixed hook cost |
| diffusers, use_stream=False | 2.2 s | Weights are **not pinned**; falls to pageable transfers (~4 GB/s) |
| **Custom (windowed prefetch)** | **0.369 s (+0.156 s)** | — |

This is not a defect in diffusers — the fixed cost of a generic hook mechanism (per-group Python work and synchronization), multiplied by 48, is simply too heavy for this workload. Estimate **fixed cost per hook × number of groups** and you will know in advance whether the generic tool suffices.

### 4.3 Designing the custom streamer (~100 lines)

The scheme is simple, but four points make it correct.

**(1) Enumerating what to move — don't forget bnb's quant_state**

`named_parameters()` and `named_buffers()` are not enough for a quantized layer. bitsandbytes' `Params4bit` carries its quantization metadata (`quant_state.absmax`, etc.) as **attribute tensors that are neither parameters nor buffers**. Miss them and you get weights on the GPU with scales on the CPU:

```python
for _, p in module.named_parameters(recurse=True):
    items.append((p, "data", p.data))
    qs = getattr(p, "quant_state", None)
    if qs is not None:
        for attr in ("absmax", "code", "offset"):
            t = getattr(qs, attr, None)
            if isinstance(t, torch.Tensor):
                items.append((qs, attr, t))
```

**(2) Synchronize with streams + events, per layer**

A dedicated copy stream transfers layer i's tensors with `non_blocking=True` and records layer i's event. On the compute side, each layer's forward_pre_hook waits **only on its own event**. Layer 0's compute and layer 1's transfer run concurrently.

**(3) Stop the allocator from recycling too early — record_stream**

Allocate on the copy stream, use on the compute stream, drop the reference when done — this pattern has a trap. PyTorch's caching allocator reuses "freed" memory immediately, but **a kernel still in flight on another stream may be reading it**. The fix: call `tensor.record_stream(compute_stream)` when use begins, so the allocator defers reuse until the compute stream has passed that point. Forget this and you get an intermittent, nearly unreproducible corruption bug.

**(4) Restoration is pointer reassignment only — zero D2H copies**

Weights never change during inference. So at the end of encoding, all you do is point `param.data` back at the saved pinned CPU tensors. **No GPU→CPU write-back ever happens.** Once you notice this, half of the classic offload round-trip cost simply does not exist.

### 4.4 The trap: full-depth prefetch defeats the purpose

Our first version enqueued all 48 layer copies up front. Speed was fine (+0.166 s) — but **at the moment of encoding, all layers (4.35 GB) were on the GPU simultaneously**, largely cancelling the residency reduction. We limited prefetch to a window (window=2): after layer i completes, release layer i and enqueue layer i+2. Extra VRAM during encode drops to two layers plus activations (~1.5 GB total). Widening the window to 4 or 8 does not change speed — the inequality in 4.1 is already satisfied at 2.

### 4.5 Verification: check bit-equality on **two consecutive runs**

Correctness is verified against the all-resident baseline — but always run the streamed encode **twice in a row** and compare both. The first run validates the streaming itself; **the second validates the restoration path** (can we stream correctly again from the pointer-restored state?). A single-run check will miss bugs that corrupt state while still producing one correct output.

---

## 5. Technique 4: Reclaim allocator fragmentation (−1.3–1.6 GB)

With techniques 1–2 applied, the narration test completed at 28.8 GB. Yet **the real application's conversation mode OOMed at the margin** — this is where synthetic single-shape tests and real applications diverge.

A real application cycles through **multiple tensor shapes**: idle clips, speaking anchors, low-resolution first chunks, main chunks. Each shape change makes the allocator carve differently sized blocks; freed blocks rarely fit the next request. The result is "reserved but unallocated: 1.33 GB" — memory that is claimed yet unusable.

The fix is one line:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It makes segments growable so fragments can be extended and reused; for multi-shape production workloads it was effectively mandatory. The bigger lesson: **validate borderline-VRAM configurations against the real application's shape distribution, not a synthetic benchmark.** The synthetic figure of 28.8 GB was correct — and production still failed.

---

## 6. Interaction with CUDA Graphs — losing once, then winning it back

CUDA Graphs (capture/replay of the denoise loop) are the speed side's workhorse and the VRAM side's troublemaker:

- Each captured shape holds a private pool that **grows monotonically and is never released**
- Every new shape triggers a capture (warmup + workspace allocation) with a **transient spike**

The symptom we hit in conversation mode: chunks whose shape was already captured succeeded, while **only the new-shape chunks failed, alternating** (a 364 MB workspace allocation OOM). This "replay-only-succeeds" pattern is characteristic of graph-pool pressure — when you see it, suspect category 3 from section 1.

At that point we concluded "no graphs at 32 GB" (the graph gain at these resolutions was marginal — 3.5 s vs 3.4 s — so it was the correct trade). But after technique 3 freed 5 GB, we re-enabled graphs and **the conversation E2E completed at a 27.1 GB peak with graphs on**. The final configuration keeps graphs enabled.

Generalized: **"the fastest configuration" and "the configuration whose memory does not grow monotonically" conflict at borderline VRAM.** When headroom drops below a few GB, choose the latter first, create room through reduction, then take the former back.

---

## 7. Measurement methodology

Half of this work was measurement. Tools and principles:

**The ballast method** — allocate a dummy tensor to cap free VRAM at the target card's size. You can reproduce "31 GB free" on a 48 GB machine without owning a 32 GB card; a ~15-line script. Every "RTX 5090-equivalent" figure in this project comes from this method. Caveat: it caps *free capacity* only — it does not reproduce fragmentation patterns or context sizes.

**Per-process peak sampling** — poll `nvidia-smi --query-compute-apps` at 0.5 s intervals and keep the max for the target PID. In-app `torch.cuda.max_memory_allocated()` sees only PyTorch-managed memory, so record both and know the gap (context etc.); in our measurements it was consistently 1.5–2 GB.

**Bit-equality checks stay within one process** — a byproduct finding: the nvfp4 stack produces **different outputs for the same seed across process restarts** (a trajectory divergence around PSNR 27 dB), while being fully deterministic within a process (framemd5-identical). Verify new features across a process boundary and you will misattribute this nondeterminism to your feature — we nearly did, and only a control experiment (same configuration, process restart only) disentangled it.

**Read the whole OOM message** — with section 1's categories, a single OOM message points at what to cut. "Tried to allocate 364 MiB / free 215 MiB / private pools 910 MB / reserved-unallocated 1.33 GB" says the culprit is pools and fragmentation, not resident weights.

---

## 8. Summary of numbers

The final configuration (`nvfp4-32gb` preset):

| Measure | Reduction | Speed cost |
|---|---|---|
| 1: Skip upsamplers | −1.2 GB resident | Zero (disables unused features) |
| 2: Embedding to CPU + logits removal | −1.9 GB resident, −0.5 GB transient | Zero (no measurable difference) |
| 3: TE layer streaming (window 2) | −5.0 GB resident | +0.16 s per encode |
| 4: expandable_segments | −1.3–1.6 GB fragments | Zero |
| Total | Resident 32.5 → **23.6 GB** | Chunk generation 3.5 → ~4.0 s (vs 4.8 s playback) |

Verified operating point (31 GB free cap, TTS on a second GPU, LLM on another host): 7 consecutive conversation chunks, all succeeded, 27.1 GB peak, CUDA Graph enabled.

## 9. Checklist for applying this elsewhere

1. Did you classify the OOM (weights / transient / pools / fragmentation)?
2. Is any module resident that is never forwarded (grep the call sites)?
3. Are there bf16 heavyweights outside quantization's reach (embeddings, norms, heads)?
4. Do you know the tied-weight relationships?
5. Table-lookup components (embeddings) are CPU candidates — but enumerate every reference path in the source first.
6. Are you computing outputs nobody reads (logits, etc.)?
7. For streaming, verify "per-layer transfer < per-layer compute" arithmetically before coding.
8. For off-the-shelf offload mechanisms, measure "fixed hook cost × group count" before adopting.
9. Did you include the quantization library's metadata tensors (quant_state etc.) in what you move?
10. Did you use record_stream to prevent premature allocator reuse?
11. Did you verify bit-equality on two consecutive runs, within one process?
12. Was the final validation done against the real application's shape distribution, not synthetic jobs?

---

All corresponding implementations are in this repository: `app/testream.py` (layer streaming), `app/tediet.py` (embedding-to-CPU + logits skip), `app/generator.py` (conditional upsampler loading), `app/config.py` (flag definitions). Everything is enabled via environment variables only, and every default is OFF (behavior identical to before).
