import os
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

import regex


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_pred_tokens(
    file: BinaryIO,
    start: int,
    end: int,
) -> dict[bytes, int]:
    """
    Split the file into tokens and count their occurrences
    """
    if start >= end:
        return {}

    file.seek(start)
    chunk = file.read(end - start).decode("utf-8", errors="ignore")
    tokens = PRED_TOKEN_PATTERN.findall(chunk)

    counts: dict[bytes, int] = {}
    for token in tokens:
        token_bytes = token.encode("utf-8")
        counts[token_bytes] = counts.get(token_bytes, 0) + 1

    return counts


PRED_TOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _count_pred_token_chunk(args: tuple[str, int, int]) -> dict[bytes, int]:
    file_path, start, end = args
    with open(file_path, "rb") as file:
        return split_pred_tokens(file, start, end)


def count_pred_tokens_parallel(
    file_path: str,
    boundaries: list[int],
    max_workers: int | None = None,
) -> dict[bytes, int]:
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if not chunk_ranges:
        return {}

    global_counts: dict[bytes, int] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for counts in executor.map(
            _count_pred_token_chunk,
            [(file_path, start, end) for start, end in chunk_ranges],
        ):
            for token, count in counts.items():
                global_counts[token] = global_counts.get(token, 0) + count

    return global_counts
