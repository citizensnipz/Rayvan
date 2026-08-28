from __future__ import annotations

import torch
from torch import Tensor, nn

from .data import CharacterTokenizer
from .model import EMCOutput


def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Tensor:
    if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
        raise ValueError("prompt_token_ids must be a non-empty one-dimensional tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if not greedy and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")

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
                    probabilities = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(0)
                generated = torch.cat((generated, next_token.reshape(1)))
    finally:
        model.train(was_training)
    return generated.cpu()


def generate_text(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    greedy: bool = False,
) -> str:
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    generated = generate_token_ids(
        model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        greedy=greedy,
    )
    return tokenizer.decode(generated.tolist())
