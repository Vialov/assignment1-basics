import heapq

from cs336_basics.bpe_tokenizer import TokenizerTrainer, pk


def test_train_skips_stale_heap_entries_and_uses_latest_version():
    trainer = TokenizerTrainer(
        file_path="unused.txt",
        dict_size=3,
        split_special_token="<|endoftext|>",
        special_tokens=[],
    )
    trainer.tokens = [b"a", b"b"]

    pair_key = pk(0, 1)
    pair_order = trainer._pair_order(pair_key)

    trainer.pair_counts = {pair_key: 3}
    trainer.pair_versions = {pair_key: 2}
    trainer.pair_heap = [
        (-10, pair_order, 1, pair_key),  # stale: wrong version and stale count
        (-3, pair_order, 2, pair_key),   # fresh: matches current version/count
    ]
    heapq.heapify(trainer.pair_heap)

    merged_pairs: list[int] = []

    def fake_merge_pair(selected_pair_key: int) -> None:
        merged_pairs.append(selected_pair_key)
        trainer.pair_heap.clear()

    trainer.merge_pair = fake_merge_pair  # type: ignore[assignment]

    trainer.train()

    assert merged_pairs == [pair_key]
