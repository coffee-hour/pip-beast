"""Sampling and checkpoint loading for Pip-Beast. Pure PyTorch, no external runner.

Generation uses a KV cache by default: one token per step, with a re-prime on the most recent
half-window when the context fills max_seq_len. Pass use_cache=False (or --no_cache) for the
original uncached path that recomputes the sliding window each token.
"""
import torch

from model import BeastConfig, PipBeast


def load_checkpoint(path: str, device: str = 'cuda'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = BeastConfig(**ckpt['config'])
    model = PipBeast(cfg)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), ckpt


def _sample(logits, temperature: float, top_k: int):
    logits = logits[:, -1, :].float() / max(temperature, 1e-5)
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


@torch.no_grad()
def stream_complete(model, tok, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
                    top_k: int = 40, device: str = 'cuda', use_cache: bool = True):
    """Yield newly generated text as it is sampled (not including the prompt)."""
    dev_type = 'cuda' if device.startswith('cuda') else 'cpu'
    dtype = torch.bfloat16 if dev_type == 'cuda' else torch.float32
    max_len = model.cfg.max_seq_len
    eos = getattr(tok, 'eos_id', None)
    ids = tok.encode(prompt)
    if not ids:
        ids = [eos if eos is not None else 0]
    out = []
    emitted = ''

    def delta(final: bool) -> str:
        nonlocal emitted
        text = tok.decode(out)
        if not final:
            text = text.rstrip('\ufffd')  # hold back an incomplete UTF-8 sequence
        if len(text) <= len(emitted):
            return ''
        piece = text[len(emitted):]
        emitted = text
        return piece

    def prefill(tokens):
        t = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.autocast(device_type=dev_type, dtype=dtype):
            logits, caches = model.forward_cached(t, None, 0)
        return logits, caches, len(tokens)

    def logits_uncached():
        window = torch.tensor([ids[-max_len:]], dtype=torch.long, device=device)
        with torch.autocast(device_type=dev_type, dtype=dtype):
            logits, _ = model(window)
        return logits

    logits = caches = None
    pos = 0
    if use_cache:
        logits, caches, pos = prefill(ids[-max_len:])

    for _ in range(max_new_tokens):
        if not use_cache:
            logits = logits_uncached()
        nxt = _sample(logits, temperature, top_k)
        token = nxt.item()
        if eos is not None and token == eos:
            break
        out.append(token)
        ids.append(token)
        piece = delta(False)
        if piece:
            yield piece
        if use_cache:
            if pos >= max_len:
                # cache is full: re-prime on the most recent half-window (includes this token)
                logits, caches, pos = prefill(ids[-(max_len // 2):])
            else:
                with torch.autocast(device_type=dev_type, dtype=dtype):
                    logits, caches = model.forward_cached(nxt, caches, pos)
                pos += 1

    tail = delta(True)
    if tail:
        yield tail


def complete(model, tok, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
             top_k: int = 40, device: str = 'cuda', use_cache: bool = True) -> str:
    """Return only the newly generated text (not the prompt)."""
    return ''.join(stream_complete(model, tok, prompt, max_new_tokens, temperature, top_k,
                                   device, use_cache))


if __name__ == '__main__':
    import argparse

    from tokenizer import BeastTokenizer

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/best.pt')
    p.add_argument('--tokenizer', default='data/tokenizer.json')
    p.add_argument('--prompt', required=True)
    p.add_argument('--max_new_tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=40)
    p.add_argument('--no_cache', action='store_true', help='use the uncached fallback path')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_checkpoint(a.ckpt, dev)
    tok = BeastTokenizer(a.tokenizer)
    print(a.prompt + complete(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k,
                              dev, use_cache=not a.no_cache))
