#!/usr/bin/env python3
import argparse
from pathlib import Path

from cs336_basics.bpe_tokenizer import count_pred_tokens_parallel, find_chunk_boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count pre-tokenized words for the first N chunks of a file.",
    )
    parser.add_argument("file_path", type=Path, help="Path to the input file.")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")
    if args.num_processes <= 0:
        raise ValueError("num_processes must be positive.")

    with args.file_path.open("rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            desired_num_chunks=max(args.num_chunks, args.num_processes),
            split_special_token=args.split_special_token.encode("utf-8"),
        )

    max_chunks = min(args.num_chunks, max(len(boundaries) - 1, 0))
    selected_boundaries = boundaries[: max_chunks + 1]

    global_counts = count_pred_tokens_parallel(
        str(args.file_path),
        selected_boundaries,
        max_workers=args.num_processes,
    )

    for token, count in sorted(global_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{token.decode('utf-8', errors='replace')}\t{count}")


if __name__ == "__main__":
    main()
