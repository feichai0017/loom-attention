"""Create a local random-weight Qwen2 model for offline engine benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=4096)
    parser.add_argument("--attention-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--stable-token-zero",
        action="store_true",
        help=(
            "keep nonzero Q/K/V projections but zero their output projection "
            "and make token 0 a deterministic generation winner"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(args.seed)
    config = Qwen2Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1.0e-6,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.loom_fixture = (
        "stable-token-zero" if args.stable_token_zero else "random-weights"
    )
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(original_dtype)
    model.eval()
    if args.stable_token_zero:
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(1.0)
            model.model.norm.weight.fill_(1.0)
            model.lm_head.weight.zero_()
            model.lm_head.weight[0].fill_(1.0)
            for layer in model.model.layers:
                layer.input_layernorm.weight.fill_(1.0)
                layer.post_attention_layernorm.weight.fill_(1.0)
                layer.self_attn.o_proj.weight.zero_()
                layer.mlp.down_proj.weight.zero_()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "output": str(args.output.resolve()),
            "parameters": parameters,
            "storage_dtype": str(next(model.parameters()).dtype),
            "stable_token_zero": args.stable_token_zero,
        }
    )


if __name__ == "__main__":
    main()
