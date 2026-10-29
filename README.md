# Pip-Beast

A standalone 1.11B-parameter decoder-only transformer, tuned to train on a single NVIDIA RTX 5090 (32 GB, Blackwell). Sovereign: no external model runners, no Ollama. Everything runs from this repo.

## Architecture

- 22 layers, dim 2048, 16 query heads, 4 KV heads (GQA), head dim 128
- SwiGLU FFN with hidden dim 5504
- RMSNorm (eps 1e-5), RoPE, tied input/output embeddings
- Vocab 32000 (byte-level BPE, trained on your data)
- Context 2048 by default, configurable up to 4096
- Attention via scaled_dot_product_attention, which uses FlashAttention kernels on supported GPUs

Satellites (rule-based, no extra weights) sit around the core: router, thalamus, endocrine, hippocampus, specialists, critic, dream (offline consolidation), orchestrator. See satellites/.

## Quick start (any terminal)

    ./setup.sh                      # venv, CUDA 12.8 PyTorch, requirements, GPU check
    ./run.sh prepare input.txt      # tokenizer + train.bin / val.bin
    ./run.sh train                  # train; auto-resumes from checkpoints/
    ./run.sh chat                   # chat with satellites
    ./run.sh chat --core            # chat with the core model only

Requirements: Linux or WSL2, Python 3.10+, NVIDIA driver 580 or newer (needed for RTX 50-series), and a CUDA-capable GPU.

## Training on the 5090

    python train_5090.py --batch 8 --accum 8 --seq_len 2048

Defaults are tuned for 32 GB: bf16 autocast, fused AdamW, cosine LR with warmup, gradient accumulation (effective batch = batch x accum x seq_len tokens), torch.compile.

Useful flags:
- --grad_ckpt: activation checkpointing, for larger batches or --seq_len 4096
- --use_8bit_adam: bitsandbytes 8-bit AdamW, to save optimizer memory (install bitsandbytes separately)
- --no_compile: disable torch.compile for debugging
- --wandb: log to Weights & Biases

Telemetry is printed every log interval: loss, LR, gradient norm, tokens per second, MFU (against the 5090 dense BF16 peak), and peak VRAM. Eval loss runs every --eval_interval steps.

Checkpoints go to checkpoints/ as step_NNNNNNN.pt (with optimizer state) plus best.pt (weights only). The last --keep_checkpoints step files are kept. To resume, run the same train command again.

## Chat and satellites

chat.py runs in any terminal. Commands: /quit, /reset (clear hormone state), /core and /sat (toggle the satellite layer).

To consolidate episodic memory between sessions:

    python -m satellites.dream checkpoints/episodes.json --keep 500

## Files

- model.py: the 1.11B architecture
- train_5090.py: training loop
- generate.py: sampling and checkpoint loading
- tokenizer.py: byte-level BPE (copied from Pip) and BeastTokenizer wrapper
- prepare_data.py: builds tokenizer and token bins
- chat.py: terminal chat
- satellites/: router, thalamus, endocrine, hippocampus, specialists, critic, dream, orchestrator
- setup.sh, run.sh: one-shot setup and runner

## Notes

- checkpoints/ and data/*.bin are generated locally and should not be committed.
- Generation currently has no KV cache, so it recomputes the window each token. Fine for chat; slower for long replies.
- Satellites are heuristics (keyword routing, lexicon sentiment). Treat them as a first pass.
- The hormone model is a sampling control signal, not a claim about biology.
- The 1.11B model is not pretrained. You need your own data to train it, and output quality depends on that data and training time.
