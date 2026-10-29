"""Sampling and checkpoint loading for Pip-Beast. Pure PyTorch, no external runner.

Note: no KV cache yet, so generation recomputes the window each token. Fine for chat, slower
for long replies.
"""
import torch

from model import BeastConfig, PipBeast


def load_checkpoint(path: str, device: str = 'cuda'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = BeastConfig(**ckpt['config'])
    model = PipBeast(cfg)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def complete(model, tok, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
             top_k: int = 40, device: str = 'cuda') -> str:
    """Return only the newly generated text (not the prompt)."""
    dev_type = 'cuda' if device.startswith('cuda') else 'cpu'
    dtype = torch.bfloat16 if dev_type == 'cuda' else torch.float32
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    eos = getattr(tok, 'eos_id', None)
    out = []
    for _ in range(max_new_tokens):
        window = idx[:, -model.cfg.max_seq_len:]
        with torch.autocast(device_type=dev_type, dtype=dtype):
            logits, _ = model(window)
        logits = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        token = nxt.item()
        if eos is not None and token == eos:
            break
        out.append(token)
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(out)


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
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_checkpoint(a.ckpt, dev)
    tok = BeastTokenizer(a.tokenizer)
    print(a.prompt + complete(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k, dev))
