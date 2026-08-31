from __future__ import annotations

import torch
from torch import Tensor, nn

from .tokenization import TextTokenizer
from .model import EMCOutput


def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    greedy: bool = False,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> Tensor:
    if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
        raise ValueError("prompt_token_ids must be a non-empty one-dimensional tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if not greedy and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    device = next(model.parameters()).device
    generated = prompt_token_ids.to(device=device, dtype=torch.long)
    maximum_context = int(model.config.max_sequence_length)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = generated[-maximum_context:].unsqueeze(0)
                output = model(context)
                logits = output.logits if isinstance(output, EMCOutput) else output
                next_logits = logits[0, -1]
                if greedy:
                    next_token = torch.argmax(next_logits)
                else:
                    filtered_logits = _filter_sampling_logits(
                        next_logits / temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    probabilities = torch.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(0)
                generated = torch.cat((generated, next_token.reshape(1)))
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
    finally:
        model.train(was_training)
    return generated.cpu()


def generate_text(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    greedy: bool = False,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    generated = generate_token_ids(
        model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        greedy=greedy,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated.tolist())


def _filter_sampling_logits(
    logits: Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> Tensor:
    filtered = logits.clone()
    if top_k is not None and top_k < filtered.numel():
        cutoff = torch.topk(filtered, k=top_k).values[-1]
        filtered[filtered < cutoff] = -torch.inf
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = -torch.inf
    return filtered
