# LTX-2.5 Diffusers Server + Web UI

English | [日本語](README.md)

A local web application for generating video with synchronized audio using `Lightricks/LTX-2.5-Diffusers`. It serves an asynchronous FastAPI job API and a web UI from the same process. It supports T2AV, I2V, first/last-frame conditioning (FLF2V), and arbitrary image/video conditions. The default high-quality path upsamples the initial latent output by 2× and applies an additional 3-step refinement pass.

## UI sample

![LTX-2.5 Studio web UI](docs/ui-sample.png)

## Requirements

- An NVIDIA GPU, a compatible driver, and sufficient RAM/VRAM
- Python 3.11 or later, or Docker with NVIDIA Container Toolkit
- `ffmpeg`
- A Hugging Face read token from an account that has accepted the terms on the [model page](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)

The model has approximately 19 billion parameters and requires substantial storage. Allow enough time and free disk space for the initial download. The default `model` offload mode reduces VRAM usage at the cost of slower inference.

## Sequential download and NF4 quantization

Run the provided script first. It avoids retaining multiple large components at once. Each component is downloaded into a dedicated cache, and that cache is removed only after the saved NF4 component or additional checkpoint has passed a reload test. An interrupted download can be resumed with the same command. Accept the terms for both the main LTX-2.5 model and the [Pixel Spatial Upscaler IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) before starting.

```bash
docker build -t ltx25-server .
docker run --rm --gpus 'device=0' \
  -v "$PWD:/app" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -w /app ltx25-server \
  python scripts/download_quantize_ltx25.py
```

By default, the script stops if less than 80 GiB is available when processing begins. It downloads the latent upsamplers and official Pixel Spatial Upscaler IC-LoRA, but not `transformer_full`, arbitrary user LoRAs, or the diffusion decoder. To add only the Pixel IC-LoRA to an existing installation, run `python scripts/download_quantize_ltx25.py --component pixel_upscaler`.

## Run with Docker

```bash
cp .env.example .env
# Set HF_TOKEN in .env
docker compose up --build
```

Open <http://localhost:8000> in a browser.

## Run in a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set HF_TOKEN in .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API specification is available at <http://localhost:8000/docs>.

## Generation modes

- `t2av`: text to video with audio
- `i2v`: one first-frame image to video with audio
- `flf2v`: interpolation with both the first and last images fixed
- `condition`: up to eight images or videos at arbitrary latent-frame positions and strengths
- `t2i`: one still image from a prompt (distilled two-stage → 2x-resolution PNG)
- `refine_image`: a 2x reinterpretation / variation still of an input image
- `ref2i`: a new-scene still that preserves the referenced identity

The web UI separates generation mode from rendering options. Landscape presets are 768×512, 768×448, 960×544, 1280×704, and 1920×1088; portrait presets transpose these dimensions to 512×768, 448×768, 544×960, 704×1280, and 1088×1920; 512×512 is also available. **2× resolution upscale** is available for T2AV, I2V, FLF2V, and Reference conditions up to a 960×544 base (544×960 portrait), producing 1920×1088 (1088×1920 portrait). Choose `Latent Upscale` for the existing 2× latent interpolation plus 3-step refinement path, or `Pixel IC-LoRA` to append the low-resolution first-stage video as reference latents and generatively synthesize detail with the official IC-LoRA. Pixel mode preserves composition, motion, and subjects but is not a pixel-faithful scaler because it invents high-frequency detail. Set `upscale_method=latent` or `pixel` through the API. The 1280×704 and larger presets are direct-generation-only because of their substantially higher VRAM requirements; selecting one disables spatial upscaling. When upscaling is disabled, generation uses a single 8-step pass. Selecting the first image in I2V or FLF2V automatically chooses a standard base orientation. The API field `quality=high/draft` remains for backward compatibility; new clients should use `upscale=true/false`.

**2× frame-rate upscale** can be enabled independently. Temporal Latent Upscale plus 3-step refinement converts 121 frames at 24 fps into 241 frames at 48 fps while preserving the video and audio duration. Set `temporal_upscale=true` through the API. When both spatial and temporal upscaling are enabled, refinement still runs only once. On an existing installation, run `scripts/download_quantize_ltx25.py --component temporal` once to download the required component.

**Retake** regenerates only a selected time range in the source video at the latent level. The source resolution and frame rate are detected automatically, and video/audio outside the selected range are preserved. You can regenerate video, audio, or both. Source videos must contain `8n+1` frames and have dimensions divisible by 32.

**Extend** adds 1–20 seconds of new video and audio to the beginning or end of a source video. The reference range is fixed as a clean-latent prefix or suffix, preserving subject, motion, and composition at the boundary. The source resolution and frame rate are adopted automatically, and reference/extension ranges are aligned to eight-frame intervals. A single generation can contain at most 481 reference-plus-extension frames.

### Still-image modes (t2i / refine_image / ref2i)

Three modes reuse the video pipeline to produce still PNGs (recipes follow the verified probes under `scratch_t2i_probe/`). Outputs are saved as `outputs/t2i_*.png` / `refine_*.png` / `ref2i_*.png` and recorded in the history DB as `image_url` (no MP4 is written). LoRA blending, seeds, progress, and the job queue work exactly as in the video modes.

- **`t2i`**: fixed `num_frames=9`, distilled two-stage (8 sigmas → 2x latent upsample → 3-sigma refine), VAE decode; the center frame is saved at twice the base resolution (default 512² → 1024²). When `decoder: "diffusion"` is requested, `num_frames` is internally promoted to 17 (logged) to satisfy the NATTEN kernel-size constraint (11×11×11 > 9 frames), and the diffusion decoder is used.
- **`refine_image`**: conditions the same two-stage recipe on an input image (registered via `/api/assets`, passed as `conditions` with `index: 0`). `strength` (0.1–1.0, default 1.0) controls how strongly the reference is preserved. Decoding is always VAE. At strength 1.0 the measured mean abs diff vs. the input is ~0.099 (matches probe P6).
- **`ref2i`**: one or more reference images (reusing the `conditions` schema with per-image latent index/strength) plus a new-scene prompt, run single-stage on the base 30-step schedule (guidance 3.0); the `frame_position: "last"` (default) or `"center"` frame is extracted. `num_frames` is selectable from 25/41/49 (default 49; larger values help when moving far from the reference). Decoding is always VAE — requesting `decoder: "diffusion"` falls back to VAE with a warning (image-conditioned diffusion decoding blurs, per the probes).

```bash
# t2i (default: 512² base → 1024² PNG)
curl -X POST http://localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"mode":"t2i","prompt":"A photorealistic portrait, golden hour light","width":512,"height":512,"seed":42}'

# refine_image (2x reinterpretation of an input image)
curl -X POST http://localhost:8000/api/jobs -H 'content-type: application/json' \
  -d "{\"mode\":\"refine_image\",\"prompt\":\"...\",\"width\":512,\"height\":512,\"strength\":1.0,\"conditions\":[{\"asset_id\":\"$ASSET_ID\",\"kind\":\"image\",\"index\":0}]}"

# ref2i (reference → new-scene still, last frame extracted)
curl -X POST http://localhost:8000/api/jobs -H 'content-type: application/json' \
  -d "{\"mode\":\"ref2i\",\"prompt\":\"The same woman, new scene...\",\"width\":512,\"height\":512,\"num_frames\":49,\"frame_position\":\"last\",\"conditions\":[{\"asset_id\":\"$ASSET_ID\",\"kind\":\"image\",\"index\":0,\"strength\":1.0}]}"
```

Measured (RTX PRO 6000 Blackwell 96GB, `OFFLOAD_MODE=model`, 512² base, seed=42): t2i 46.5 s including the first model load (the generation itself takes ~6–8 s once loaded) with 17.9 GB peak VRAM; t2i + diffusion decoder ~30 s warm; refine_image 30.0 s warm; ref2i at nf=49 45.5 s warm (~34 s denoise). In the web UI the three modes appear in the mode selector, results render as PNG tiles in the gallery (downloadable), and a "use this image as the I2V/FLF2V first frame" button drops a generated PNG straight into the I2V first-image field.

**Audio → Video** encodes WAV, MP3, M4A, FLAC, OGG, or AAC through the Audio VAE and generates only the video modality while keeping the audio latent fixed. You can select an audio start point and up to 20 seconds of audio, with an optional first-frame image. The original input waveform, rather than VAE-reconstructed audio, is used in the output.

The web UI also supports duration-head automatic length selection, a Native Multishot shot editor, and persistent history with session numbers. History is stored in `outputs/history.sqlite3` by default. Because output length is unknown before automatic-duration generation, Reference conditions are limited to the beginning (0%) or end (100%) in this mode.

Place `.safetensors` LoRAs directly under `loras/`, then use **Reload** in the web UI. Up to four adapters can be combined per job, with individual strengths from -2.0 to 2.0. Adapters are unloaded after every job, including failed jobs, so their settings never carry over. Use LoRAs compatible with the LTX-2/LTX-2.5 Diffusers transformer. Older LTX-Video LoRAs or weights with ComfyUI-specific keys may require conversion.

**IC-LoRA / Reference** is a dedicated mode separate from standard Reference conditions. Select one IC-LoRA and one reference-sheet image or video. Reference latents are concatenated to the generation sequence as additional tokens. An image is internally repeated into a static reference video matching the output frame count. The generic path is stage 1 only, uses a fixed duration, and recommends 768×448. Specialized IC-LoRAs requiring extra inputs—such as HDR scene embeddings or DubIt audio references—are not supported by this generic path.

The LoRA listing API reads `model_version` and `reference_downscale_factor` from safetensors metadata. The current generic IC-LoRA mode accepts only adapters with a reference downscale factor of 1 and rejects other adapters in the UI before generation.

**Optimize for LTX-2.5 with AI** uses an OpenAI-compatible `/chat/completions` endpoint. Set `LLM_BASE_URL` and `LLM_MODEL` in `.env`, plus `LLM_API_KEY` when required. Standard generation remains available when the enhancer is not configured.

Select a mode and upload conditioning files through the web UI. API clients must upload each asset first, then use the returned 32-character `id` in the generation request.

```bash
# I2V
ASSET_ID=$(curl -s -F 'file=@first.png' http://localhost:8000/api/assets \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -X POST http://localhost:8000/api/jobs \
  -H 'content-type: application/json' \
  -d "{\"mode\":\"i2v\",\"prompt\":\"The camera slowly moves forward.\",\"conditions\":[{\"asset_id\":\"$ASSET_ID\",\"kind\":\"image\",\"index\":0,\"strength\":1.0}]}"
```

For FLF2V, upload two images and set the first condition to `index: 0` and the last condition to `index: -1`. In general condition mode, use `kind: video` for video assets.

```bash
curl -X POST http://localhost:8000/api/jobs \
  -H 'content-type: application/json' \
  -d '{"prompt":"A white crane flies over rainy Tokyo. Cinematic camera. Distant thunder.","seed":42}'
```

Use the returned `id` with `GET /api/jobs/{id}`. Once the job reaches `completed`, download the MP4 from `video_url`. Jobs run one at a time to prevent concurrent GPU workloads from exhausting VRAM.

## Configuration

- `OFFLOAD_MODE=model`: recommended; CPU offload by model component
- `OFFLOAD_MODE=sequential`: lowest VRAM usage and slowest execution
- `OFFLOAD_MODE=none`: keep every model on the GPU; intended for high-VRAM systems
- `MODEL_REVISION`: pinned by default to a verified commit for reproducibility
- `MAX_QUEUE_SIZE`: number of waiting jobs; default 4
- `HISTORY_DB`: SQLite file for sessions and generation history; default `outputs/history.sqlite3`
- `LLM_BASE_URL`: OpenAI-compatible `/v1` base URL used for prompt rewriting
- `LLM_MODEL`: external LLM model name
- `LLM_API_KEY`: external LLM API key; may be empty for a local endpoint
- `INPUT_DIR`: uploaded conditioning assets; default `inputs`
- `LORA_DIR`: LoRA `.safetensors` directory; default `loras`
- `MAX_UPLOAD_SIZE_MB`: maximum size per uploaded file; default 500 MB
- `LTX25_DECODER`: decoder after 2× latent upscale. `diffusion` uses the high-quality diffusion decoder and adds about 18 seconds with NATTEN; `vae` uses the faster convolutional VAE. The request-level `decoder` field overrides this setting. Jobs without 2× upscaling always use VAE decoding.
- `LTX25_VIDEO_CRF`: libx264 CRF for output MP4 files; default 18. Applied to every path and decoder. This uses a higher bitrate than the former effective default of approximately CRF 23, reducing compression-related detail loss.
- `LTX25_TRANSFORMER_PRECISION`: `nf4` for the default bitsandbytes 4-bit transformer, `fp8` for the bf16 release weights compressed in place with layerwise casting (fp8_e4m3fn storage / bf16 compute), or `bf16` for the approximately 38 GB release weights. `fp8` matches bf16 quality while keeping measured peak VRAM at 26.5 GB for stills and 28.9 GB for 121-frame video, making it the recommended mode for 48 GB-class GPUs (the cast is applied on the CPU, so there is no transient ~38 GB GPU-side peak; requires the ~38 GB bf16 transformer shards in the HF cache). `bf16` is intended for 96 GB-class GPUs; keep `nf4` on 24 GB-class GPUs. The text encoder remains NF4 in every mode. **`nvfp4` (added 2026-09, sm_120 Blackwell only):** runs the official Lightricks Blackwell-native FP4 distilled transformer (~19 GB resident) directly through `torch._scaled_mm` FP4 GEMM (`app/nvfp4.py`; raw GEMM 3.2-3.8× faster than bf16 — the fastest configuration for realtime use. Set `LTX25_NVFP4_CKPT` to a local file, otherwise it is fetched from the HF Hub automatically).

- `LTX25_CUDA_GRAPH`: set to `1` to capture/replay the entire transformer forward as a CUDA Graph, eliminating CPU kernel-launch overhead during denoising (`app/cudagraph.py`). Output is **bit-identical** to eager. Requires `OFFLOAD_MODE=none` (ignored with a warning otherwise). Jobs using LoRA automatically fall back to eager. The gain grows at small resolutions × few steps (see "Realtime generation" below).
- `LTX25_CUDA_GRAPH_MAX_CAPTURES`: maximum number of captured shapes (default 8). One graph is captured per (resolution, frame count, fps, mode) combination; **shapes beyond the limit silently fall back to eager** after a warning log. Raise this for multi-shape workloads.
- `LTX25_LOAD_UPSAMPLERS`: `0` skips loading the latent/temporal upsamplers (−1.2 GB GPU-resident). Intended for realtime workloads (always `upscale: false`); with `0`, upscale/temporal_upscale/t2i-quality requests fail with a clear error (default `1`).
- `LTX25_TE_DIET`: `1` moves the Gemma text encoder's embedding table (bf16, 1.88 GB — bnb 4-bit only quantizes Linear layers, embeddings stay bf16) to the CPU with a gather bridge, and skips the unused all-token logits computation (~0.5 GB transient) (`app/tediet.py`). Total: −1.9 GB resident + −0.5 GB transient, with near-zero speed impact. Requires `OFFLOAD_MODE=none` (warned and ignored otherwise; default `0`). See "Realtime on 32 GB-class GPUs" below.
- `LTX25_COMPILE_BLOCKS`: [experimental, not recommended] per-block torch.compile (see the docstring in `app/compileblocks.py`). Wins in isolated probes but showed no end-to-end gain on the server and regresses at small resolutions, hence default `off`.
- `LTX25_VIDEO_ENCODER`: `nvenc` (default, h264_nvenc) or `x264`. Falls back to x264 automatically when NVENC is unavailable.
- `LTX25_NVENC_PRESET`: NVENC preset (`p1` fastest to `p7` highest quality, default `p7`). Use `p4` for realtime workloads to shave 0.1-0.15 s off encoding.
- `LTX25_DECODE_SINGLE_TILE`: diffusion-decoder tiling policy (`auto` default / `on` / `off`). Uses a single tile when free VRAM allows (~1.23× faster, seam-free).
- `LTX25_STAGE_DEBUG`: set to `1` to log per-stage timings (text encode / denoise / decode / encode). Default 0.

## Measured quality and performance

RTX PRO 6000 Blackwell 96 GB, 512×512 base to 1024² output, 121 frames, seed 42:

| Configuration | Total time | Peak VRAM | Notes |
|---|---:|---:|---|
| NF4 + VAE decode + CRF 18 (default, FastAPI, model offload) | 76 s | — | Backward-compatible path; audio mux verified |
| NF4 + diffusion decoder (NATTEN na3d, torch 2.11+cu130, FastAPI, model offload) | ~90 s | 17.3 GB during decode | **18.0 s decode**; quality matches the flex path below |
| NF4 + diffusion decoder (legacy flex-attention, torch 2.9, all models on GPU, first run) | 329 s | 61.9 GB | 302 s decode including flex kernel compilation |
| NF4 + diffusion decoder (legacy flex, warm second run in the same process) | 314 s | 61.9 GB | 293 s decode; compilation accounts for only about 10 s |
| BF16 transformer + diffusion decoder (legacy flex, all models on GPU) | 321 s | 87.3 GB | Fits on a 96 GB GPU; unsuitable for a 24 GB GPU |

**The diffusion decoder preserves substantially more fine detail than the default VAE decoder.** Fine-texture retention in smooth regions (minimum variance over 128 px patches) improved from 1.67 to 2.41. VAE-specific false grain-like high-frequency noise also disappeared; the decrease in global Laplacian variance from 36.3 to 25.2 reflects that noise reduction.

**NATTEN kernel (introduced 2026-08-19):** the project was updated to torch 2.11.0+cu130 and applies `LTX2VideoVaeNeighborhoodNattenProcessor` using the prebuilt `shi-labs/natten` na3d kernel obtained through the `kernels` package (torch211-cxx11-cu130, verified on sm_120). The implementation is in `app/generator.py`. If the kernel is unavailable, it falls back to compiled flex-attention and reports the selected path at startup. In a decode-only measurement using `scratch_ab/latents.pt` at 1024²×121 frames, runtime improved from **293 s (warm flex) to 18.3 s (about 16× faster)** and peak VRAM decreased from 35.8 GB to 17.3 GB. Quality measurements matched the flex path: Laplacian variance 25.13 vs. 25.17, minimum smooth-region 128 px patch variance 2.399 vs. 2.414, and mean absolute raw-frame difference 0.066/255. With decoding reduced to about 18 seconds, **`diffusion` is now the default decoder** (`LTX25_DECODER=vae` restores the legacy path).

## Realtime generation and acceleration (measured 2026-09)

### Purpose and limitations (read this first)

The **goal of this acceleration stack is minimizing latency for serving/realtime workloads — small resolutions × few steps × repeated identical settings** (generation faster than playback). Use it with these limitations in mind:

- **It barely helps large-resolution, many-step quality generation.** CUDA Graph gains concentrate where CPU kernel launches dominate (-20% at 384×288, -4% at 512×288); at 1024×576+, 8 steps, or the 2x-upscale path, GPU execution dominates and the gain is a few percent.
- **Incompatible with low-VRAM operation.** CUDA Graph requires `OFFLOAD_MODE=none` (everything resident); it cannot be combined with model/sequential offload (the 24-48 GB-class memory-saving configurations). Memory saving and minimum latency are separate axes.
- **Only nvfp4 is a verified combination.** `fp8` (layerwise casting) + CUDA Graph is unverified (interaction with the casting hooks).
- **Jobs using LoRA automatically fall back to eager** (no graph benefit).
- **Every new shape (resolution / frame count / fps / mode) pays a one-time capture cost (+1.5-2 s)** — one-off generation with ever-changing settings cannot amortize it.
- torch.compile on top showed no measured gain (see the compile section).
- Model-level quality limits (the fps=16 periodic wobble, long-clip stalls) are not solved by acceleration — see the quality notes below.

With nvfp4 + CUDA Graph + NVENC combined, **streaming generation faster than playback (realtime ratio < 1.0x)** is achievable. All measurements below: RTX PRO 6000 Blackwell 96 GB (sm_120), distilled sigmas, 4 steps, ~5-second clips, t2av, `LTX25_TRANSFORMER_PRECISION=nvfp4 OFFLOAD_MODE=none LTX25_CUDA_GRAPH=1 LTX25_NVENC_PRESET=p4`.

### Realtime on 32 GB-class GPUs (RTX 5090) — measured 2026-09-27

The all-resident configuration needs ~32.5 GB of free VRAM just to load its weights, so it does not fit 32 GB-class cards (measured: OOM during weight loading with free VRAM capped at 31 GB). Adding `LTX25_LOAD_UPSAMPLERS=0` and `LTX25_TE_DIET=1` brings **resident/peak down to 28.8 GB, generating 512×384 / 20 fps / 97-frame / 4-step a2v chunks in a steady 3.8 s (0.79x realtime)** — verified with a VRAM-capped ballast test at 31 GB free (headless RTX 5090 equivalent). Speed is essentially identical to the 48 GB configuration (3.5 s), and output is fully deterministic within a process (identical framemd5 for a fixed seed).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LTX25_TRANSFORMER_PRECISION=nvfp4 OFFLOAD_MODE=none LTX25_CUDA_GRAPH=1 \
LTX25_NVENC_PRESET=p4 LTX25_LOAD_UPSAMPLERS=0 LTX25_TE_DIET=1 ./run.sh
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required for multi-shape workloads. A real application (idle clips, speaking anchors, low-res first chunks, and main chunks each trigger their own CUDA Graph capture) accumulates 1.3-1.6 GB of "reserved but unallocated" allocator fragmentation, which causes borderline OOM at exactly 31 GB free (measured). With expandable_segments reclaiming it, a full 7-chunk continuous narration run completes without OOM (3.2-3.7 s per 4.8 s chunk = 0.66-0.77x realtime). Verified compatible with CUDA Graph capture.

Prerequisites: (1) upscale and the t2i quality path are unavailable; (2) headroom is only ~2 GB, so **the GPU must be dedicated to this server** — run TTS/LLM workloads on the CPU or another host; (3) headless operation (display on another GPU / iGPU).

### CUDA Graph gains (bit-identical, measured)

| Condition (4 steps) | no graph | with graph |
|---|---|---|
| 512×288×121f | 2.48 s | 2.38 s (-4%) |
| 384×288×81f | 2.30-2.37 s | **1.83-1.88 s (-20%)** |

The smaller the resolution, the more CPU-launch-bound the denoise becomes, so the gain grows. Video framemd5 and audio md5 match eager exactly. Each new shape pays a one-time capture cost (+1.5-2 s).

### Realtime resolution ceilings (~5-second clips, 4 steps, 1.0x realtime budget)

| fps | safe (≤0.90x) | edge (~0.98x) |
|---|---|---|
| 20 fps (97f) | 768×512 | 768×576 |
| 24 fps (121f) | 704×480 | 768×480 |

At 8 steps the ceiling roughly halves (~340k pixels at 16 fps). a2v (audio-conditioned) adds only ~+0.15 s on average over t2av (watch boundary cells only).

### Important: fps=16 has a quality defect

Requesting fps=16 produces a **structural motion wobble with a 4.0-second (64-frame) period** (brief stalls / slight rewind feel). LTX-2.5 is trained mostly at 24/25 fps, so the RoPE time coordinate goes out of distribution at 16 fps; an A/B with identical seed/content confirmed the artifact disappears at 20 fps and 24 fps. **Use 20 fps for realtime work** (clean quality with a wider budget than 24 fps).

### Long single-shot generation

At 704×416, 8 steps, 16 fps, generation succeeds up to the API limit (`num_frames <= 481`, 30.06 s) — a 30-second clip generates in 29.4 s at 40.9 GB peak VRAM. However, **seed-dependent "motion stall then jump" events appear sporadically in clips of ~25 s and longer**. Use `probes/detect_motion_stall.py` (~0.5 s per video, exit code 1 on detection) as a post-generation QC gate and retry with a different seed when it fires.

### probes/

Verification scripts used during development are included: CUDA Graph equivalence/speed (`probe_cudagraph*.py`), the torch.compile investigation record (`probe_compile_*.py`; conclusion: no production gain), nvfp4 quantization checks (`probe_nvfp4_*.py`), and the motion-stall detector (`detect_motion_stall.py`).

## License

The application code originally implemented in this repository is provided under the [Apache License 2.0](LICENSE).

> [!IMPORTANT]
> The Apache License 2.0 does not apply to LTX model weights, LTX-derived LoRAs or checkpoints, Gemma models, or other third-party components.

LTX-2/LTX-2.5 and derivatives are governed by the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE). It includes use restrictions and redistribution obligations. Commercial use by entities with annual revenue of USD 10 million or more requires a paid commercial license from Lightricks. Review the latest original agreement before using or distributing the model, LoRAs, or generated output. See [LTX Model Licensing](https://ltx.io/model/license) for commercial licensing information.

Third-party models, libraries, and kernels—including the Gemma text encoder—remain subject to the licenses and terms published by their respective providers.

`.env` and generated assets are excluded from Git. For public deployment, add authentication, TLS, and rate limiting at the reverse proxy.
