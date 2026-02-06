import os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO
from collections import defaultdict
import heapq

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
    special_tokens: list[str] | None = None,
    *,
    pattern: regex.Pattern | None = None,
) -> dict[bytes, int]:
    """
    Split the file into tokens and count their occurrences
    """
    if start >= end:
        return {}

    file.seek(start)
    chunk = file.read(end - start).decode("utf-8", errors="ignore")
    if pattern is None:
        pattern = get_pred_token_pattern(special_tokens)
    tokens = pattern.findall(chunk)

    counts: dict[bytes, int] = {}
    for token in tokens:
        token_bytes = token.encode("utf-8")
        counts[token_bytes] = counts.get(token_bytes, 0) + 1

    return counts


BASE_PRED_TOKEN_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


@lru_cache(maxsize=128)
def _compile_pred_token_pattern(special_tokens_key: tuple[str, ...]) -> regex.Pattern:
    if not special_tokens_key:
        return regex.compile(BASE_PRED_TOKEN_PATTERN)

    deduped = sorted(set(special_tokens_key), key=lambda s: (-len(s), s))
    escaped = [regex.escape(token) for token in deduped]
    specials_pattern = "|".join(escaped)
    return regex.compile(f"{specials_pattern}|{BASE_PRED_TOKEN_PATTERN}")


_PRED_TOKEN_PATTERN = regex.compile(BASE_PRED_TOKEN_PATTERN)


def get_pred_token_pattern(special_tokens: list[str] | None = None) -> regex.Pattern:
    if not special_tokens:
        return _PRED_TOKEN_PATTERN
    normalized = [token.decode("utf-8") if isinstance(token, bytes) else token for token in special_tokens]
    return _compile_pred_token_pattern(tuple(normalized))


def init_pred_token_pattern(special_tokens: list[str] | None) -> None:
    global _PRED_TOKEN_PATTERN
    _PRED_TOKEN_PATTERN = get_pred_token_pattern(special_tokens)


def _count_pred_token_chunk(args: tuple[str, int, int]) -> dict[bytes, int]:
    file_path, start, end = args
    with open(file_path, "rb") as file:
        return split_pred_tokens(file, start, end, pattern=_PRED_TOKEN_PATTERN)


def count_pred_tokens_parallel(
    file_path: str,
    boundaries: list[int],
    max_workers: int | None = None,
    special_tokens: list[str] | None = None,
) -> dict[bytes, int]:
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if not chunk_ranges:
        return {}

    global_counts: dict[bytes, int] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_pred_token_pattern,
        initargs=(special_tokens,),
    ) as executor:
        for counts in executor.map(
            _count_pred_token_chunk,
            [(file_path, start, end) for start, end in chunk_ranges],
        ):
            for token, count in counts.items():
                global_counts[token] = global_counts.get(token, 0) + count

    return global_counts


def pk(a: int, b: int) -> int:
    return (a << 32) | b


def unpack(k: int) -> tuple[int,int]:
    return (k >> 32), (k & 0xFFFFFFFF)


class TokenizerTrainer:
    def __init__(
            self,
            file_path: str,
            dict_size: int,
            split_special_token: str,
            special_tokens: list[str] | None = None
    ):
        self.file_path = file_path
        self.special_tokens = special_tokens
        self.split_special_token = split_special_token
        self.dict_size = dict_size

        self.merges: list[tuple[bytes, bytes]] = [] # List of merges performed, in order (using token bytes)

        self.tokens: list[bytes] = []

        self.words: list[bytes] = []
        self.word_counts: list[int] = [] # Count of each word (using word indices)
        self.word_heads: list[int] = []

        # Token positions and mappings
        self.token_list: list[int] = []  # List of token occurrences
        self.alive_tokens: list[bool] = [] # Whether each token is still alive (not merged)
        self.next_token: list[int] = []
        self.prev_token: list[int] = []
        self.next_token: list[int] = []
        self.token_to_word: list[int] = [] # Maps each token to the index of the word it belongs to

        self.pair_counts: dict[int, int] = {} # Counts of each token pair (using token indices)
        self.pair_versions: dict[int, int] = {} # For each pair, the version of the tokens when the pair was last updated
        self.pair_occurrences: dict[int, list[int]] = defaultdict(list) # For each pair, the list of token indices where this pair occurs
        self.pair_heap: list[tuple[int, int, int]] = [] # Heap of pairs to consider for merging, as (negative count, version, pair_key)

    def initial_count(self, num_chunks: int = 16, max_workers: int = 4) -> None:
        self.tokens = [bytes([i]) for i in range(256)]

        # Step 1: Split the file into words and count occurrences
        with open(self.file_path, "rb") as file:
            boundaries = find_chunk_boundaries(
                file,
                desired_num_chunks=num_chunks,
                split_special_token=self.split_special_token.encode("utf-8"),
            )

        print(f"Found {len(boundaries)} chunks.")
        max_chunks = min(num_chunks, max(len(boundaries) - 1, 0))
        selected_boundaries = boundaries[: max_chunks + 1]

        global_counts = count_pred_tokens_parallel(
            self.file_path,
            selected_boundaries,
            max_workers=max_workers,
            special_tokens=self.special_tokens,
        )

        # Step 2: Build initial word list and token list
        for word, count in global_counts.items():
            word_ind = len(self.words)
            self.words.append(word)
            self.word_counts.append(count)
            self.word_heads.append(len(self.token_list))

            prev_token = -1
            for token in word:
                token_ind = len(self.token_list)
                self.token_list.append(token)
                self.alive_tokens.append(True)
                self.token_to_word.append(word_ind)

                if prev_token != -1:
                    self.next_token.append(token_ind)
                    pair_key = pk(prev_token, token)
                    self.pair_counts[pair_key] = self.pair_counts.get(pair_key, 0) + count
                    self.pair_occurrences[pair_key].append(token_ind-1) # Store the index of the first token in the pair
                    self.prev_token.append(token_ind - 1)
                else:
                    self.prev_token.append(-1) # No previous token for the first token in the word

                prev_token = token

            self.next_token.append(-1) # End of word

        self.pair_versions = {pair_key: 0 for pair_key in self.pair_counts.keys()}
        self.pair_heap = [(-count, 0, pair_key) for pair_key, count in self.pair_counts.items()]
        heapq.heapify(self.pair_heap)