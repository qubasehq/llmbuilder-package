"""
Example: Train a small GPT model on cybersecurity text files using LLMBuilder.

Usage:
  python docs/train_model.py --data_dir ./Model_Test --output_dir ./Model_Test/output \
      --prompt "Cybersecurity is important because" --epochs 5

If --data_dir is omitted, it defaults to the directory containing this script.
If --output_dir is omitted, it defaults to <data_dir>/output.

This script uses small-friendly settings (block_size=64, batch_size=1) so it
works on tiny datasets. It trains, saves checkpoints, and performs a sample
text generation from the latest/best checkpoint.
"""
from __future__ import annotations

import argparse

import llmbuilder
from llmbuilder import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train and generate with LLMBuilder on small text datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory with .txt files (default: folder of this script)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save outputs (default: <data_dir>/output)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size (small data friendly)",
    )
    parser.add_argument(
        "--block_size", type=int, default=64, help="Context window size for training"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Model embedding dimension"
    )
    parser.add_argument(
        "--layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Cybersecurity is important because",
        help="Prompt for sample generation",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=80, help="Tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling top_p"
    )
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(args.data_dir)
    # If a nested Data/ folder exists, prefer it
    if (data_dir / "Data").exists():
        data_dir = data_dir / "Data"
    output_dir = Path(args.output_dir) if args.output_dir else (data_dir / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Configs mapped to llmbuilder expected keys
    config = {
        # tokenizer/dataset convenience
        "vocab_size": 8000,
        "block_size": int(args.block_size),
        # training config -> llmbuilder.config.TrainingConfig
        "training": {
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "num_epochs": int(args.epochs),
            "max_grad_norm": 1.0,
            "save_every": 1,
            "log_every": 10,
        },
        # model config -> llmbuilder.config.ModelConfig
        "model": {
            "embedding_dim": int(args.embed_dim),
            "num_layers": int(args.layers),
            "num_heads": int(args.heads),
            "max_seq_length": int(args.block_size),
            "dropout": 0.1,
        },
    }

    print("Starting LLMBuilder training pipeline...")
    pipeline = llmbuilder.train(
        data_path=str(data_dir),
        output_dir=str(output_dir),
        config=config,
        clean=False,
    )

    # Generation
    best_ckpt = output_dir / "checkpoints" / "best_checkpoint.pt"
    latest_ckpt = output_dir / "checkpoints" / "latest_checkpoint.pt"
    model_ckpt = best_ckpt if best_ckpt.exists() else latest_ckpt
    tokenizer_dir = output_dir / "tokenizer"

    if model_ckpt.exists() and tokenizer_dir.exists():
        print("\nGenerating sample text with trained model...")
        text = llmbuilder.generate_text(
            model_path=str(model_ckpt),
            tokenizer_path=str(tokenizer_dir),
            prompt=args.prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        print("\nSample generation:\n" + text)
    else:
        print("\nSkipping generation because artifacts were not found.")


if __name__ == "__main__":
    main()
