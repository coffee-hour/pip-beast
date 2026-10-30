# Pip-Beast

A standalone decoder-only transformer, tuned to train on a single NVIDIA RTX 5090 (32 GB, Blackwell). Sovereign: no external model runners, no Ollama. Everything runs from this repo.

## Presets

| preset | params (tied embeddings) | layers | dim | query / KV heads | head dim | SwiGLU hidden |
|---|---|---|---|---|---|---|
| fast | about 222M (221,545,472) | 16 | 1024 | 8 / 4 | 128 | 2816 |
| beast (default) | about 1.04B (1,040,279,552) | 22 | 2048 | 16 / 4 | 128 | 5504 |

Both presets use vocab 32000, context 2048 by default, RMSNorm (eps 1e-5), RoPE, tied input/output embeddings, and GQA. Run `./run.sh presets` to print the counts from the code.

Note: the beast count is about 1.04B with tied embeddings. The untied count is about 1.11B. Earlier docs said 1.11B; that was the untied figure.

## Architecture

- RMSNorm, RoPE, SwiGLU FFN, grouped-query attention
- Attention via scaled_dot_product_attention (FlashAttention kernels on supported GPUs)
- KV cache for generation (see below)

Satellites (rule-based, no extra weights) sit around the core: router, hippocampus, specialists, critic, dream (offline consolidation), orchestrator. See satellites/. Experimental work, including the feelings and chemistry satellites, lives in the separate private repo pip-future.

## Quick start

    ./setup.sh                          # venv, CUDA 12.8 PyTorch, requirements, GPU check
    ./run.sh prepare input.txt          # tokenizer + train.bin / val.bin
    ./run.sh train --preset fast        # or --preset beast (default); auto-resumes
    ./run.sh ui                         # web UI at http://localhost:8000
    ./run.sh chat                       # terminal chat with satellites
    ./run.sh chat --core                # terminal chat, core model only

Requirements: Linux or WSL2, Python 3.10+, NVIDIA driver 580 or newer (needed for RTX 50-series), and a CUDA-capable GPU.

## Training

    ./run.sh train --preset fast
    ./run.sh train --preset beast
    python train_5090.py --preset beast --target_tokens 1048576

What the preset sets:
- Model shape (table above)
- Peak learning rate: fast 6e-4, beast 3e-4
- Tokens per optimizer step: 524,288 for both
- Warmup: max(100, 1% of --max_steps)

Auto batch: unless you pass --batch, the trainer probes the largest power-of-two micro-batch that fits in VRAM (budget 92% of VRAM, reserving room for AdamW state). It then computes gradient accumulation so each optimizer step covers about --target_tokens tokens. The plan is printed at startup. Use --no_auto_batch with --batch and --accum to pin values. The old form still works:

    python train_5090.py --batch 8 --accum 8 --no_auto_batch --seq_len 2048

Useful flags:
- --grad_ckpt: activation checkpointing, for larger batches or --seq_len 4096
- --use_8bit_adam: bitsandbytes 8-bit AdamW (install bitsandbytes separately)
- --no_compile: disable torch.compile for debugging
- --wandb: log to Weights & Biases
- --out_dir: use a separate checkpoint directory per preset (a checkpoint from one preset cannot resume into the other)

Telemetry is printed every log interval: loss, LR, gradient norm, tokens per second, MFU (against the 5090 dense BF16 peak), and peak VRAM. Eval loss runs every --eval_interval steps.

Checkpoints go to checkpoints/ as step_NNNNNNN.pt (with optimizer state) plus best.pt (weights only). The last --keep_checkpoints step files are kept. To resume, run the same train command again.

Training times: not measured yet. Throughput depends on the GPU, driver, and data, so run a short job and read tok/s from the log to estimate time to a target token count.

## Generation and the KV cache

Generation uses a KV cache: each new token runs one forward step against cached keys and values, instead of recomputing the whole window. When the context fills max_seq_len, the cache is rebuilt from the most recent half-window. The original uncached path is still available with `python generate.py --no_cache --prompt ...`, and `complete(..., use_cache=False)` in Python.

## Web UI

    ./run.sh ui                         # http://localhost:8000 (this machine only)
    python web_ui.py --host 0.0.0.0     # expose on the LAN
    python web_ui.py --core             # start with satellites off

The UI is a single page (web/index.html) served by web_ui.py, which uses only the Python standard library (http.server, no extra packages). Tokens stream over Server-Sent Events. The page has:
- temperature, top-k, and max-token sliders (these apply to the core model)
- a satellites toggle
- reset, which clears the conversation

Satellite replies are produced whole by the orchestrator, then shown in chunks, so they do not stream token by token. The web UI serializes generation with one lock because there is one model on one GPU.

The UI loads JetBrains Mono from Google Fonts. Without network access it falls back to the system monospace font.

## Chat and satellites

chat.py runs in any terminal. Commands: /quit, /reset (clear conversation), /core and /sat (toggle the satellite layer).

To consolidate episodic memory between sessions:

    python -m satellites.dream checkpoints/episodes.json --keep 500

## Files

- model.py: architecture, PRESETS, and the cached decode path (forward_cached)
- train_5090.py: training loop, presets, auto batch sizing
- generate.py: sampling (stream_complete and complete), KV cache, checkpoint loading
- web_ui.py: standard-library HTTP server for the UI on port 8000
- web/index.html: the UI page
- tokenizer.py: byte-level BPE (copied from Pip) and BeastTokenizer wrapper
- prepare_data.py: builds tokenizer and token bins
- chat.py: terminal chat
- satellites/: router, hippocampus, specialists, critic, dream, orchestrator
- setup.sh, run.sh: one-shot setup and runner
- requirements.txt: torch, numpy, safetensors (web UI adds nothing)

## Notes

- checkpoints/ and data/*.bin are generated locally and should not be committed.
- Satellites are heuristics (keyword routing, lexicon sentiment). Treat them as a first pass.
- The models are not pretrained. You need your own data to train them, and output quality depends on that data and training time.
