#!/usr/bin/env python3
import argparse
from pathlib import Path

from cs336_basics.bpe_tokenizer import count_pred_tokens_parallel, find_chunk_boundaries


def parse_args() -> argparse.Namespace:
    DEFAULT_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
    DEFAULT_CHUNKS = 10
    DEFAULT_SPLITS = 40
    DEFAULT_SPECIAL_TOKENS = "<|endoftext|>"

    parser = argparse.ArgumentParser(
        description="Count pre-tokenized words for the first N chunks of a file.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to the input file.",
        default=DEFAULT_PATH,
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_CHUNKS,
        help="Number of initial chunks to process.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes for counting.",
    )
    parser.add_argument(
        "--split-special-token",
        type=str,
        default="<|endoftext|>",
        help="Special token used to align chunk boundaries.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=DEFAULT_SPECIAL_TOKENS,
        help="Special tokens to preserve as single pre-tokens.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=DEFAULT_SPLITS,
        help="Desired number of splits when finding chunk boundaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")
    if args.num_processes <= 0:
        raise ValueError("num_processes must be positive.")

    with args.path.open("rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            desired_num_chunks=args.num_splits,
            split_special_token=args.split_special_token.encode("utf-8"),
        )

    print(f"Found {len(boundaries)} chunks.")
    max_chunks = min(args.num_chunks, max(len(boundaries) - 1, 0))
    selected_boundaries = boundaries[: max_chunks + 1]

    global_counts = count_pred_tokens_parallel(
        str(args.path),
        selected_boundaries,
        max_workers=args.num_processes,
        special_tokens=args.special_tokens,
    )

    for token, count in sorted(global_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{token.decode('utf-8', errors='replace')}\t{count}")


if __name__ == "__main__":
    main()
