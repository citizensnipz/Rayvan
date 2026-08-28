import torch

from rayvan_emc import EMCConfig, EMCModel, EMCOutput


def main() -> None:
    torch.manual_seed(7)
    config = EMCConfig(
        latent_dim=16,
        num_modules=6,
        modules_per_cycle=2,
        num_cycles=3,
        vocab_size=64,
        module_hidden_dim=32,
        attention_heads=2,
    )
    model = EMCModel(config).eval()
    token_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        result = model(token_ids, return_trace=True)

    if not isinstance(result, EMCOutput):
        raise RuntimeError("trace-enabled forward pass did not return EMCOutput")

    print(f"Token IDs shape: {tuple(token_ids.shape)}")
    print(f"Output logits shape: {tuple(result.logits.shape)}")
    for cycle in result.trace:
        weights = [round(weight, 4) for weight in cycle.router_weights.tolist()]
        print(
            f"Cycle {cycle.cycle}: modules {list(cycle.selected_modules)} "
            f"weights {weights} latent {cycle.latent_shape}"
        )


if __name__ == "__main__":
    main()
