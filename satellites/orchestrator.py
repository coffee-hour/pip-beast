"""Orchestrator: wires the satellites to Pip-Beast. The model code is not modified.

Flow per turn:
  router -> thalamus -> endocrine (tick) -> hippocampus recall -> specialist (if routed)
  -> generate with hormone-shaped sampling -> critic (one cooler retry) -> hippocampus store

Usage (repo root):
    from model import PipBeast  # via generate.load_checkpoint
    from generate import load_checkpoint
    from tokenizer import BeastTokenizer
    from satellites.orchestrator import Orchestrator
    model, _ = load_checkpoint('checkpoints/best.pt', 'cuda')
    pip = Orchestrator(model, BeastTokenizer('data/tokenizer.json'), memory_path='checkpoints/episodes.json')
    print(pip.respond('hello there'))
"""
import torch

from generate import complete
from satellites import critic, router, specialists, thalamus
from satellites.endocrine import Endocrine
from satellites.hippocampus import Hippocampus


class Orchestrator:
    def __init__(self, model, tok, memory_path=None, device=None, max_new_tokens=200):
        self.model = model
        self.tok = tok
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.endo = Endocrine()
        self.hip = Hippocampus(memory_path)
        self.max_new_tokens = max_new_tokens

    def _specialist_note(self, route: str, text: str) -> str:
        if route == 'math':
            try:
                return f'[math] {text.strip()} = {specialists.math_eval(text)}'
            except (ValueError, SyntaxError, ZeroDivisionError):
                return ''
        if route == 'code':
            r = specialists.python_check(text)
            return '' if r['ok'] else f"[code] syntax issue: {r['error']}"
        return ''

    def respond(self, user_text: str) -> str:
        route = router.classify(user_text)
        stim = thalamus.perceive(user_text)
        self.endo.tick(stim.valence, stim.arousal, stim.threat)
        temp, top_k = self.endo.sampling()

        memories = self.hip.recall(user_text, k=2)
        note = self._specialist_note(route, user_text)

        parts = []
        if memories:
            parts.append('Memory: ' + ' | '.join(memories))
        if note:
            parts.append(note)
        parts.append(f'User: {user_text}')
        prompt = '\n'.join(parts) + '\nPip:'

        reply = complete(self.model, self.tok, prompt, self.max_new_tokens, temp, top_k, self.device).strip()
        verdict = critic.evaluate(reply)
        if not verdict.ok:
            retry = complete(self.model, self.tok, prompt, self.max_new_tokens,
                             max(0.4, temp - 0.2), top_k, self.device).strip()
            if critic.evaluate(retry).score > verdict.score:
                reply = retry

        self.hip.store(f'User: {user_text} | Pip: {reply}', salience=stim.arousal)
        return reply
