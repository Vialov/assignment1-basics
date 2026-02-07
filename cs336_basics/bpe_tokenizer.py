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
        pattern = get_pred_token_pattern()

    counts: dict[bytes, int] = {}
    if special_tokens:
        special_token_pattern = regex.compile(
            "|".join(regex.escape(token) for token in sorted(set(special_tokens), key=lambda s: (-len(s), s)))
        )
        chunks = special_token_pattern.split(chunk)
    else:
        chunks = [chunk]

    for segment in chunks:
        for token in pattern.findall(segment):
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


def get_pred_token_pattern() -> regex.Pattern:
    return _PRED_TOKEN_PATTERN


def init_pred_token_pattern(special_tokens: list[str] | None) -> None:
    global _PRED_TOKEN_PATTERN
    _PRED_TOKEN_PATTERN = get_pred_token_pattern()


def _count_pred_token_chunk(args: tuple[str, int, int, list[str]]) -> dict[bytes, int]:
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as file:
        return split_pred_tokens(file, start, end, pattern=_PRED_TOKEN_PATTERN, special_tokens=special_tokens)


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
            [(file_path, start, end, special_tokens) for start, end in chunk_ranges],
        ):
            for token, count in counts.items():
                global_counts[token] = global_counts.get(token, 0) + count

    return global_counts


def pk(a: int, b: int) -> int:
    return (a << 32) | b


def unpack(k: int) -> tuple[int,int]:
    return (k >> 32), (k & 0xFFFFFFFF)


class PairOrder:
    __slots__ = ("left", "right")

    def __init__(self, left: bytes, right: bytes) -> None:
        self.left = left
        self.right = right

    def __lt__(self, other: "PairOrder") -> bool:
        if not isinstance(other, PairOrder):
            return NotImplemented
        # Reverse lexicographic order so lexicographically greater pairs win ties.
        return (self.left, self.right) > (other.left, other.right)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PairOrder):
            return False
        return (self.left, self.right) == (other.left, other.right)


class TokenizerTrainer:
    def __init__(
            self,
            file_path: str,
            dict_size: int,
            split_special_token: str,
            special_tokens: list[str] | None = None
    ):
        self.file_path = file_path
        self.special_tokens = special_tokens or []
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
        self.pair_heap: list[tuple[int, PairOrder, int, int]] = [] # Heap of pairs to consider for merging, as (negative count, pair_order, version, pair_key)

        self._touches: dict[int, int] = {}  # Temporary dict to track which pairs need to be updated after a merge, mapping token index to the last version it was touched

    def _pair_order(self, pair_key: int) -> PairOrder:
        token_A, token_B = unpack(pair_key)
        return PairOrder(self.tokens[token_A], self.tokens[token_B])

    def initial_count(self, num_chunks: int = 16, max_workers: int = 4) -> None:
        self.tokens = [bytes([i]) for i in range(256)]

        # Step 1: Split the file into words and count occurrences
        with open(self.file_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            global_counts = split_pred_tokens(
                file,
                0,
                file_size,
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
        self.pair_heap = [(-count, self._pair_order(pair_key), 0, pair_key) for pair_key, count in self.pair_counts.items()]
        heapq.heapify(self.pair_heap)

    def merge_pair(self, pair_key: int) -> None:
        self._touches = {}

        token_A, token_B = unpack(pair_key)
        new_token = self.tokens[token_A] + self.tokens[token_B]
        new_token_ind = len(self.tokens)
        self.tokens.append(new_token)

        for index_A in self.pair_occurrences[pair_key]:
            # Check if the pair is still valid (both tokens are alive and in the correct order)
            if (not self.alive_tokens[index_A]) or (self.token_list[index_A] != token_A):
                continue

            index_B = self.next_token[index_A]
            if index_B == -1 or self.token_list[index_B] != token_B or (not self.alive_tokens[index_B]):
                continue

            self.token_list[index_A] = new_token_ind
            self.alive_tokens[index_B] = False
            prev_index = self.prev_token[index_A]
            next_index = self.next_token[index_B]
            word_count = self.word_counts[self.token_to_word[index_A]]

            if prev_index != -1:
                prev_token = self.token_list[prev_index]
                prev_pair_key = pk(prev_token, token_A)
                self._touches[prev_pair_key] = self._touches.get(prev_pair_key, 0) - word_count

                new_pair = pk(prev_token, new_token_ind)
                self._touches[new_pair] = self._touches.get(new_pair, 0) + word_count
                self.pair_occurrences[new_pair].append(prev_index)

            self.next_token[index_A] = next_index
            if next_index != -1:
                next_token = self.token_list[next_index]
                next_pair_key = pk(token_B, next_token)
                self._touches[next_pair_key] = self._touches.get(next_pair_key, 0) - word_count

                new_pair = pk(new_token_ind, next_token)
                self._touches[new_pair] = self._touches.get(new_pair, 0) + word_count
                self.pair_occurrences[new_pair].append(index_A)
                self.prev_token[next_index] = index_A

        self.merges.append((self.tokens[token_A], self.tokens[token_B]))
        self.pair_counts[pair_key] = 0
        self.pair_versions[pair_key] += 1
        self.pair_occurrences[pair_key] = []

        for touched_pair_key, delta in self._touches.items():
            if delta == 0:
                continue
            self.pair_counts[touched_pair_key] = self.pair_counts.get(touched_pair_key, 0) + delta
            self.pair_versions[touched_pair_key] = self.pair_versions.get(touched_pair_key, 0) + 1
            heapq.heappush(
                self.pair_heap,
                (
                    -self.pair_counts[touched_pair_key],
                    self._pair_order(touched_pair_key),
                    self.pair_versions[touched_pair_key],
                    touched_pair_key,
                ),
            )

    def train(self) -> None:
        merge_count = 0
        while len(self.tokens) + len(self.special_tokens) < self.dict_size and self.pair_heap:
            neg_count, _, version, pair_key = heapq.heappop(self.pair_heap)
            actual_version = self.pair_versions[pair_key]
            actual_count = self.pair_counts[pair_key]
            if actual_count <= 0:
                continue

            if version != actual_version or actual_count != -neg_count:
                continue
            merge_count += 1
            print(f"{merge_count}: merging pair {self._pair_order(pair_key).left} + {self._pair_order(pair_key).right} with count {actual_count}")
            self.merge_pair(pair_key)

        special_tokens = [token.encode("utf-8") for token in self.special_tokens]
        self.tokens.extend(special_tokens)
