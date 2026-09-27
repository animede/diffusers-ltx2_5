# Design Notes on Making Generative AI Inference Fast

This document is not about LTX-2.5's end result ("real-time video generation") but about the acceleration techniques used to get there, organized so they transfer to other models and inference servers.

The target implementation is a 22B video+audio DiT, but the core ideas are general. Instead of lumping acceleration together as "making the GPU faster," we decompose it into separate bottlenecks — number of executions, matrix math, kernel launches, decoding, data conversion, encoding — and apply a different tool to each.

Numbers are as of 2026-09-10, measured on an RTX PRO 6000 Blackwell Workstation 96 GB (sm_120), PyTorch 2.11.0+cu130, diffusers Git version, single GPU. Absolute times depend on your environment; the subject of this document is where each technique helped and where it did not.

The low-VRAM companion to this document — the techniques that fit the same 22B model into a 32 GB-class GPU — is [Low-VRAM Techniques for Generative AI Inference](lowvram-techniques.en.md).

## 1. First, separate what you are reducing

Inference time decomposes roughly as:

```text
Total latency
  = preprocessing
  + denoise count × time per transformer forward
  + latent decode
  + pixel-format conversion and GPU→CPU transfer
  + video encode and audio mux
  + synchronization, queue waits, other fixed costs
```

The techniques adopted here do not optimize the same place twice:

| Technique | What it directly reduces | Measured effect | Main constraints |
|---|---|---:|---|
| Distilled-schedule subsampling | Number of transformer executions | ~31% faster going 8→4 steps | Requires quality evaluation |
| NVFP4 | Per-layer GEMM time and weight bandwidth | GEMM 3.2–3.8×, whole layer ~1.8× | Blackwell sm_120 only |
| CUDA Graph | CPU-side kernel-launch cost | ~20% E2E on small shapes | Requires fixed shapes, resident weights |
| NATTEN | Neighborhood attention in the diffusion decoder | Decode 293 s → 18.3 s | Needs matching kernels and input conditions |
| GPU-side uint8 conversion | CPU conversion and transfer volume | Cuts post-processing fixed cost | Watch rounding order |
| NVENC + async encode | MP4 creation on the critical path | ~0.5 s of post-processing | Most useful with back-to-back jobs |

Do not multiply the ratios in this table together. Each technique targets a different segment, and speeding one segment up makes the next one the bottleneck. Always evaluate at four levels: single kernel, one forward, per-stage time, and E2E time.

## 2. The algorithm side: reduce the execution count itself

Before any low-level optimization, the most natural question is how many times you call the same giant model. This implementation picks 4 points, evenly spaced and always keeping the first and last, from the official distilled 8-point sigma schedule:

```python
idx = [round(i * (len(sigmas) - 1) / (steps - 1)) for i in range(steps)]
picked = [sigmas[i] for i in idx]
```

The implementation is `_subsample_distilled_sigmas()` in `app/generator.py`. Going from 8 steps to 4 cut ~31%.

The important caveat: halving the computation does not halve E2E time — text encoding, VAE decode, and encoding remain as fixed costs. Also, you must select from the trajectory the distilled model was trained for, not mechanically thin out an arbitrary scheduler.

### What to verify

- A/B compare composition, motion, and audio sync at identical seeds
- Check not just still-frame quality but inter-frame velocity changes and stalls
- Record both the denoise segment and E2E time
- Split "fast" and "quality" settings per use case

This is an approximation-introducing optimization, so the acceptance criterion is fitness for purpose, not bit-equality.

## 3. The math side: shorten GEMM with NVFP4

The transformer's dominant cost is the Linear layers' matrix multiplications. This implementation loads the official NVFP4-quantized checkpoint and calls `torch._scaled_mm` — Blackwell-native block-scaled FP4 GEMM — directly. Weights are ~18.7 GB; activations are dynamically FP4-quantized per forward with a custom Triton kernel.

The flow:

```text
bf16 activation
  → dynamic quantization per 16 elements
  → FP4 packed values + FP8 block scales
  → convert scales to cuBLAS blocked layout
  → torch._scaled_mm
  → apply global scale and bias
  → bf16 output
```

GEMM alone was 3.2–3.8× over bf16; the whole layer, including quantization and scale handling, ~1.8×. When evaluating low precision, measure the whole layer — quantization, padding, layout conversion — not just theoretical FLOPS or the GEMM.

### A format is not determined by dtype alone

The most dangerous class of bug here was tensors whose shape and dtype were correct while the *meaning* was not:

- The checkpoint's packed weights put the high nibble first — the reverse of cuBLAS's packing convention
- `weight_scale` was stored already swizzled into cuBLAS's blocked layout
- Applying the usual swizzle again double-converted it
- Passing block scales in row-major order to `_scaled_mm` returned wrong numbers, not an exception

These bugs hide if you compare your own dequant against your own GEMM under the same misinterpretation — both are wrong in the same way. We finally pinned them down by direct comparison against the official bf16 weights.

When porting a quantization format, verify independently:

1. Value encoding and nibble order
2. The meaning of block scales and the order global scale is applied
3. The physical layout of the scale tensor
4. The quantized Linear alone vs. the official high-precision Linear
5. Error not just after one layer but after all blocks

Implementation: `app/nvfp4.py`; verification: `probes/probe_nvfp4_*.py`.

## 4. The execution side: batch kernel launches with CUDA Graph

Once GEMM is fast, the time spent issuing many small kernels from CPU to GPU becomes relatively visible. The GPU got faster, so launches — not GPU math — become the bottleneck.

CUDA Graph captures a sequence of GPU work once and replays it as a single issue thereafter. This implementation captures the entire diffusers transformer forward, not individual blocks:

```text
First time:
  allocate static inputs → warm up on a side stream → clear workspaces → capture

Afterwards:
  copy_ into static inputs → graph.replay() → consume static outputs immediately
```

Measured at 4 steps:

| Shape | Eager | CUDA Graph | Delta |
|---|---:|---:|---:|
| 512×288, 121 frames | 2.48 s | 2.38 s | −4% |
| 384×288, 81 frames | 2.30–2.37 s | 1.83–1.88 s | −20% |

The smaller the shape, the bigger the effect. On large shapes the GPU math is long enough to hide launch cost; on small shapes the CPU cannot issue fast enough and gaps open in the GPU's execution queue. CUDA Graph is not "always X% faster" — it helps only in launch-bound regimes.

### Preconditions

- No `.item()`, CPU-dependent branching, or CPU tensor creation inside the forward
- Shapes, dtypes, and non-tensor arguments distinguishable by the capture key
- The model's and weights' GPU addresses never change
- No model offloading
- Never replay across operations that mutate weights or the forward (e.g., LoRA)
- Static outputs consumed before the next replay
- The same shape repeats enough to amortize the 1.5–2 s initial capture cost

This implementation falls back to eager on unknown arguments, capture failure, or exceeding the capture-count limit. Good for service continuity — but for performance measurement it silently mixes eager into your numbers, so check the logs and replay counters.

Implementation: `app/cudagraph.py`; verification: `probes/probe_cudagraph.py` and `probes/probe_cudagraph_gputime.py`.

## 5. Specialized kernels: pick what matches the computation's structure

In the high-resolution path's diffusion decoder, a generic flex-attention implementation of neighborhood attention dominated. Replacing it with NATTEN's prebuilt `na3d` kernel cut decode from 293 s to 18.3 s and peak VRAM from 35.8 GB to 17.3 GB.

This is the largest single-segment improvement in this document — and the reason is not "because we compiled it." The computation is inherently local neighborhood attention, and we had been running it through something more general. Choosing a specialized kernel that matches the semantics of the operation is worth considering before dtype changes or graphing.

Quality was compared via Laplacian variance, patch variance in smooth regions, and raw frame differences, confirming practical equivalence with the old path. When swapping in a faster kernel, evaluate on three axes: speed, VRAM, and numerics/image quality.

Specialized kernels come with constraints — supported GPUs, PyTorch/CUDA ABI, kernel sizes, minimum input shapes. This implementation falls back to compiled flex-attention when the kernel cannot be fetched or loaded, and logs which path was selected at startup.

## 6. The data path: move conversion, transfer, and encoding off the critical path

Once denoise shrinks to a few seconds, 0.1–0.6 s of post-processing stops being negligible. Three measures:

### Convert to uint8 on the GPU

Instead of transferring float VAE output to the CPU and converting in NumPy, clamp, multiply by 255, round, and cast to uint8 on the GPU before transferring. This cuts both CPU-side conversion and transfer volume.

One catch: multiplying by 255 while still in bf16 did not reproduce the old path's rounding. Converting to float32 first makes framemd5 match the legacy NumPy path exactly:

```python
video = (
    video.permute(0, 1, 3, 4, 2)
    .float()
    .clamp(0, 1)
    .mul(255)
    .round()
    .to(torch.uint8)
    .cpu()
    .numpy()
)
```

### Move work to NVENC

H.264 goes through `h264_nvenc` rather than libx264. To keep the quality criterion consistent, CRF-equivalent values are mapped to VBR constant-quality. If NVENC is unavailable, or codec open fails under VRAM pressure, it retries with libx264 after removing partial outputs.

### Overlap encoding with the next denoise

MP4 encoding is handed to a dedicated single worker thread while the main generation worker proceeds to the next job's denoise. This is a throughput optimization for back-to-back submission, not a way to signal one job's completion earlier: jobs stay `running` until the encode finishes, and `completed` is never returned before the file exists.

Implementation: `app/encoding.py`; the GPU conversion and deferred-encode assembly are in `app/generator.py`; completion management in `app/jobs.py`.

## 7. Why `torch.compile` was not adopted

After CUDA Graph removed the CPU launches, we also examined fusing the small in-block kernels with Inductor. To hide the FP4 dtype from Inductor, the NVFP4 Linear was made opaque as a `torch.library.custom_op`, and we got all the way to `fullgraph=True`.

In an isolated transformer probe, compiled + graph improved one forward from 250 ms to 168 ms. But in production E2E it was a wash at the same shape, and on the small realtime shape (384×288×81 frames) it *regressed* from 1.83 s to 2.39 s — for tiny tensors, the fixed cost of the generated Triton kernels exceeded eager's ATen kernels.

Also, without `dynamic=False`, the second shape captured a dynamic generic kernel and degraded further to 2.77 s. And although a single compiled block differed by only ~0.9999 cosine, after 48 blocks the trajectory difference accumulated to ~0.965 per forward.

The lessons are crisp:

- Compile success is not speedup success
- A win in an isolated probe is not a win in E2E
- Watch dynamic-shape recompiles and the kernels actually generated
- Tiny numeric differences accumulate in deep iterative models
- Keep rejected optimizations quarantined behind an experimental flag, with the conditions for re-testing written down

Implementation: `app/compileblocks.py`; reproduction probes: `probes/probe_compile_*.py`. The production default is off.

## 8. Measure correctly

CUDA is asynchronous: the time a Python function returns is not when the GPU finished. We did in fact misread callback intervals in the denoise loop as "1.82 s → 0.48 s, 3.8×" — that was host-side issue time. GPU time measured with CUDA Events was 250 ms → 225 ms per forward, and the E2E gain was ~20% on small shapes.

Measure at these levels:

| Level | What | How |
|---|---|---|
| Kernel / op | GEMM, quantization, attention | CUDA Events, sufficient warmup |
| Module | Linear layer, one transformer forward, decoder | CUDA Events + synchronize |
| Stage | text encode, denoise, decode, encode | Segment logs; make sync points explicit for GPU segments |
| E2E | API accept to artifact complete | Wall clock; separate first run from steady state |

And include these in the measurement conditions:

- Cold start or warm
- Whether initial compile/capture is included
- Shape, frame count, fps, step count, generation mode
- Precision, offload, LoRA presence
- Whether encode completion is included
- Where the synchronization points were placed
- Whether comparisons used identical seeds

Beyond speed, use framemd5 / audio md5, cosine, image-quality metrics, and visual A/B as appropriate to the technique. For an optimization that should be bit-exact, "looks the same" is far too weak; for approximations like quantization or step reduction, "bits differ" is expected.

## 9. Quick decision table

| Situation | First technique to consider | Judgment to avoid |
|---|---|---|
| Denoise count dominates | Distillation, step reduction | Arbitrary thinning without quality checks |
| Linear/GEMM dominates, capable GPU available | Native low-precision GEMM | Judging checkpoint format by shapes alone |
| GPU utilization gaps on small shapes | CUDA Graph | Variable shapes, one-shot generation, forcing coexistence with offload |
| One attention is pathologically slow | An operation-specific kernel | Pushing through with generic compile alone |
| Post-decode is relatively heavy | GPU-side conversion, NVENC | Measuring only denoise and stopping |
| Jobs arrive back-to-back | Async encode | Completion notification before the file exists |
| Only the compile probe is fast | Re-measure E2E and quality | Enabling it in production as-is |

This project's fastest configuration is specialized for serving: models resident in ample VRAM, a small number of fixed shapes repeated. For low-VRAM operation, variable shapes, frequent LoRA swapping, or one-shot high-quality generation, the optimum changes.

## Extra: cut VRAM while keeping the speed, and widen the GPUs you can run on

The same decompose-first mindset applies to VRAM reduction. The fastest configuration — all-resident, fixed shapes — demands ample VRAM, but merely "loading only what you actually use" can drop the requirement one GPU tier. This implementation trimmed ~3.6 GB of residency with the following three measures, making single-GPU realtime work on a 32 GB-class card (28.8 GB resident measured with free VRAM capped at 31 GB, speed nearly identical to the 48 GB configuration):

1. **Don't load components you don't use.** Two upscalers that the target use case (realtime) never calls sat unconditionally resident at 1.2 GB. Skip the load behind a config flag and return a clear error if requested.
2. **Giant embedding tables can live on the CPU.** An LLM-style text encoder's vocabulary embedding (1.9 GB in bf16) is a table lookup, not math. Replace the module's forward with a bridge — indices to CPU, gathered result back to GPU — and each request costs a few MB of transfer with measured speed impact of roughly zero. Easy to miss: even in a 4-bit-quantized model, embeddings stay unquantized in bf16.
3. **Suspect unnecessary output-head computation.** Although only hidden states are used when running as a text encoder, `ForConditionalGeneration`'s forward computed all-tokens × all-vocabulary logits (0.5 GB transient) every call. A thin wrapper that bypasses lm_head removes it.

Two cautions. When placing embeddings on the CPU, check the source for hidden reference paths such as "comparison against embedding vectors" (the `inputs_embeds`-passing design broke on exactly this, and we switched to module-forward replacement). And when the remaining headroom is small, do not co-locate other workloads (TTS, LLM, etc.) on that GPU.

The full treatment of this topic — including the layer-streaming technique that goes further (−5 GB) and re-enables CUDA Graph at 32 GB — is in [Low-VRAM Techniques for Generative AI Inference](lowvram-techniques.en.md).

## 10. Principles from this implementation

1. Accelerate each bottleneck in its own layer.
2. After each speedup, re-measure what became the bottleneck.
3. Verify hardware-native formats down to physical layout, not just dtype.
4. If fixed shapes and fixed addresses can be operational requirements, CUDA Graph is strong.
5. Don't optimize an operation with a generic compiler when a specialized kernel exists.
6. On an asynchronous GPU, never confuse host time with device time.
7. Adopt only when all four align: isolated benchmark, E2E speed, reproducibility, quality.
8. Keep failed optimizations too — they are results that map the applicability boundary.

For LTX-2.5's results themselves — realtime attainment, operable resolutions, fps-specific quality issues — see the [full acceleration report](acceleration-report-20260910.md) (Japanese).
