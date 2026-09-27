# Real-Time Talking Characters from a 22B Video Diffusion Model — on a Single 32 GB GPU, with Plain Diffusers

*How we took LTX-2.5 (a 22B audio+video DiT) from 90 seconds per clip to faster-than-playback, and then fit the whole thing into a 32 GB-class GPU without giving the speed back.*

<video controls src="https://github.com/user-attachments/assets/de1c4714-00a0-4342-a8cc-ed683af00a8b" style="max-width: 100%;"></video>

*An unedited, real-speed recording: a local LLM streams a reply, local TTS voices it, and LTX-2.5 generates the talking video in 5-second chunks faster than they play back. This exact recording was made with free VRAM capped at 31 GB — a headless RTX 5090 equivalent. The character speaks Japanese; the pipeline is language-agnostic.*

Everything is open: the [video server + acceleration code](https://github.com/animede/diffusers-ltx2_5) (Apache-2.0), the [conversation app](https://github.com/animede/Realtime_Narration_Video) (Apache-2.0), and two detailed technical write-ups this article condenses — [Design Notes on Making Generative AI Inference Fast](https://github.com/animede/diffusers-ltx2_5/blob/main/docs/optimization-techniques.en.md) and [Low-VRAM Techniques for Generative AI Inference](https://github.com/animede/diffusers-ltx2_5/blob/main/docs/lowvram-techniques.en.md).

## Why plain diffusers?

Most local video-generation work happens in node-based UIs, which are wonderful for experimentation. But our target was different: a **resident server** that generates fixed-shape 5-second clips back to back, forever — a serving workload, not an interactive one. That flips the engineering priorities:

- Weights stay at fixed GPU addresses → CUDA Graph capture becomes possible
- Shapes repeat thousands of times → per-shape setup costs amortize to zero
- The pipeline boundary is ours → we can measure and cut every segment

So we built directly on `diffusers` + FastAPI and treated the pipeline as a system to be profiled, not a black box.

## Part 1: Getting to real time (the speed stack)

No single trick did this. Four layers, each attacking a *different* bottleneck:

| Layer | Technique | What it cuts | Measured |
|---|---|---|---:|
| Algorithm | Subsample the distilled sigma schedule 8→4 steps | Number of transformer forwards | −31% |
| Math | NVFP4 — Blackwell-native FP4 GEMM via `torch._scaled_mm` | Per-layer GEMM time + weight bandwidth | GEMM 3.2–3.8×, layer ~1.8× |
| Execution | CUDA Graph over the whole transformer forward | CPU kernel-launch cost | −20% E2E on small shapes, bit-identical |
| Data path | GPU-side uint8 conversion, NVENC, async encode | Post-processing on the critical path | ~0.5 s/clip |

The result on a 96 GB workstation card: **1.83 s for a ~5-second clip** at the realtime operating point (0.37× realtime ratio). The full conversation pipeline (LLM → TTS → video) delivers its first video ~2.6 s after you press Enter.

Three findings we think generalize:

**CUDA Graph helps exactly where GPUs got "too fast."** After NVFP4, the GPU finished kernels faster than the CPU could launch them — a launch-bound regime. Graphs cut small-shape latency by 20% while staying *bit-identical* to eager (we verify with framemd5 on video and md5 on audio). On large shapes the gain shrinks to ~4%: the GPU math hides the launches anyway. Graphs are not "always X% faster"; they are a targeted fix for launch-bound regions.

**Quantization formats are not defined by dtype.** The official NVFP4 checkpoint stores packed weights with the *reverse* nibble order from cuBLAS's convention, and stores `weight_scale` already swizzled into cuBLAS's blocked layout. Feed `_scaled_mm` a wrong-layout scale tensor and you get **wrong numbers, not an exception** — and if you compare your dequant against your GEMM under the same misreading, both are wrong the same way and the bug hides. Only direct comparison against the official bf16 weights caught it.

**`torch.compile` won the microbenchmark and lost production.** With the FP4 linear wrapped as a `custom_op` we reached `fullgraph=True`, and an isolated probe improved 250→168 ms per forward. In production E2E, the same configuration was a wash at large shapes and *regressed* 1.83→2.39 s at the realtime shape — Inductor's generated-kernel fixed costs exceed eager ATen on tiny tensors. It ships disabled, behind a flag, with the re-test conditions documented. Negative results are results.

## Part 2: Fitting it into 32 GB (without the classic trade)

The all-resident realtime configuration needs ~32.5 GB just to load. A 32 GB card OOMs before serving a single request. The textbook answer — model CPU offload — works (22 GB peak) but the weights cross PCIe on every job: **15 s per 5-second chunk, 3× too slow for realtime**. We wanted the VRAM reduction *without* that trade.

### Read the OOM message like a map

PyTorch's OOM message contains a full anatomy of your VRAM if you read all of it:

```
Tried to allocate 364.00 MiB ... 30.87 GiB in use. Of the allocated memory
29.06 GiB is allocated by PyTorch, with 910.24 MiB allocated in private pools
(e.g., CUDA Graphs), and 1.33 GiB is reserved by PyTorch but unallocated.
```

That is five different problems — resident weights, transient workspaces, CUDA-Graph private pools (which grow monotonically per captured shape and are never released), allocator fragmentation, and non-PyTorch overhead — and each needs a different fix. "Just quantize more" addresses only the first.

### The four cuts

**1. Don't load what you never call (−1.2 GB).** Two upsampler models sat resident that the realtime path never forwards (`upscale: false`, always). Gate them behind a flag; fail loudly if requested. Five minutes of grep for 1.2 GB.

**2. Quantization's blind spots (−1.9 GB resident, −0.5 GB transient).** "The text encoder is already 4-bit" was a false comfort: **bitsandbytes quantizes Linear layers only.** The 262k × 3840 embedding table sat on the GPU in bf16 — 1.88 GB. But an embedding is a table lookup, not math, so we bridged the module's forward: indices go to the CPU, the gathered 7.5 MB comes back. Bit-identical output, zero measurable slowdown.

Bonus: used as a text encoder, only hidden states are read — yet `ForConditionalGeneration.forward` computed all-token logits (1024 × 262k, ~0.5 GB transient) on every call. A thin wrapper that skips `lm_head` removes it.

One design lesson: our first, "cleaner" approach — precomputing `inputs_embeds` — broke, because the model internally detects multimodal tokens by *comparing against embedding vectors*, a hidden reference path that dies when the table lives on the CPU. Replacing the module's forward keeps every internal call site working. When you relocate a weight, enumerate every reference to it first.

**3. Windowed layer streaming (−5.0 GB, the main event).** The NF4 language-model body (5.7 GB) is used by every request, so it can't be dropped — but its 48 layers execute *in order*. We keep them in pinned host memory and stream them through the GPU during encoding.

Do the arithmetic before writing code: compute per layer ≈ 4.4 ms; transfer per layer (120 MB, pinned, PCIe) ≈ 2.4 ms. **Transfer < compute**, so a 2-layer prefetch window hides the copies completely. If that inequality flips on your hardware, streaming will be slow no matter how well you implement it.

We tried diffusers' `apply_group_offloading` first — always measure the existing tool — and got +1.03 s per encode: with streams it forces one layer per group, and 48 groups × ~22 ms of per-hook fixed cost is the whole overhead. The custom streamer (~100 lines) costs **+0.156 s** instead:

- a dedicated copy stream + one CUDA event per layer; each layer's pre-hook waits only on its own event
- `tensor.record_stream(compute_stream)` on use, so the caching allocator can't recycle memory a still-running kernel is reading (forget this and you get an unreproducible corruption bug)
- **bitsandbytes' `quant_state` tensors (`absmax` etc.) must move too** — they are neither parameters nor buffers, and missing them leaves weights on the GPU with scales on the CPU
- restoration is pointer reassignment to the saved pinned tensors — inference never mutates weights, so **zero device-to-host copies ever happen**
- verify bit-equality on **two consecutive runs**: the first validates streaming, the second validates the restoration path

And one trap: our first version prefetched all 48 layers up front — great latency, but all 4.35 GB sat on the GPU at once, defeating the purpose. The window matters.

**4. Allocator fragmentation (−1.3–1.6 GB).** Our synthetic single-shape test passed at 28.8 GB; the real conversation app OOMed at the margin. Real apps cycle through many shapes (idle clips, speaking anchors, low-res first chunks, main chunks), and the allocator fragments — "reserved but unallocated: 1.33 GB." One line fixes it: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The bigger lesson: **validate borderline-VRAM configs against the real application's shape distribution, not a benchmark.**

### Losing to CUDA Graphs once, then winning them back

Midway, conversation mode showed a distinctive failure: chunks whose shape was already captured succeeded; only new-shape chunks failed, alternating. That "replay-only-succeeds" pattern is the signature of graph-pool pressure. We disabled graphs at 32 GB — the honest trade at the time — and after layer streaming freed 5 GB, re-enabled them. Final state: **graphs on, conversation E2E at a 27.1 GB peak.**

### The scoreboard

| Measure | Reduction | Speed cost |
|---|---|---|
| Skip unused upsamplers | −1.2 GB | zero |
| Embedding→CPU + logits skip | −1.9 GB (+0.5 GB transient) | zero |
| Windowed layer streaming | −5.0 GB | +0.16 s/encode |
| expandable_segments | −1.3–1.6 GB fragments | zero |
| **Total** | **32.5 → 23.6 GB resident** | chunk gen 3.5 → ~4.0 s (vs 4.8 s playback) |

Verified end to end with free VRAM capped at 31 GB (headless RTX 5090 equivalent), TTS on a second GPU, LLM on another host: a 7-chunk conversation turn, every chunk generated faster than it plays, 27.1 GB peak, CUDA Graphs enabled. And it holds up outside the lab: a community user has since reported conversation mode running on a **physical RTX 5090** with this configuration. Fun fact: our test card actually has *less* raw compute than an RTX 5090 — the constraint was never speed, only VRAM.

## Measurement rules we now refuse to break

1. **GPU time comes from CUDA Events, never host timestamps.** We once misread async callback intervals as a 3.8× speedup; the real number was 1.10×.
2. **Bit-equality checks run twice, inside one process.** Cross-process comparison is invalid here: the nvfp4 stack is fully deterministic within a process (framemd5-identical) but produces *different* outputs for the same seed across process restarts (~27 dB PSNR trajectory divergence — kernel selection differs per process). We nearly blamed a new feature for this.
3. **The ballast method**: cap free VRAM with a dummy allocation to emulate a smaller card. Every "RTX 5090-equivalent" number above comes from a 15-line script on a 48 GB card.
4. **Read the whole OOM message.** It usually names the culprit for you.

## What's in the repos

- [`animede/diffusers-ltx2_5`](https://github.com/animede/diffusers-ltx2_5) — the server: NVFP4 loader (`app/nvfp4.py`), CUDA Graph wrapper (`app/cudagraph.py`), TE diet (`app/tediet.py`), layer streamer (`app/testream.py`), probes for every claim above. Every optimization is an env flag, every default is off.
- [`animede/Realtime_Narration_Video`](https://github.com/animede/Realtime_Narration_Video) — the conversation app: sentence-level TTS chunking pipelined against generation, idle-clip pools, turn continuity.
- Full write-ups: [speed](https://github.com/animede/diffusers-ltx2_5/blob/main/docs/optimization-techniques.en.md) / [low-VRAM](https://github.com/animede/diffusers-ltx2_5/blob/main/docs/lowvram-techniques.en.md).

Hardware for the numbers above: RTX PRO 6000 / PRO 5000 Blackwell (sm_120), PyTorch 2.11+cu130, diffusers Git. Application code is Apache-2.0; model weights follow Lightricks' LTX license.

If you try this on your own model — the checklists at the end of both write-ups are the distilled version of everything here. Questions and refutations welcome; every number in this article has a probe script behind it.
