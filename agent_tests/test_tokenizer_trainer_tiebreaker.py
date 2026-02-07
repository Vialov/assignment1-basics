from cs336_basics.bpe_tokenizer import TokenizerTrainer, pk


def test_train_tiebreaker_prefers_lexicographically_greater_pair(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ab ac", encoding="utf-8")

    trainer = TokenizerTrainer(
        file_path=str(file_path),
        dict_size=257,
        split_special_token="<|endoftext|>",
        special_tokens=[],
    )
    trainer.initial_count(num_chunks=1, max_workers=1)

    pair_ab = pk(ord("a"), ord("b"))
    pair_space_a = pk(ord(" "), ord("a"))
    pair_ac = pk(ord("a"), ord("c"))

    assert trainer.pair_counts[pair_ab] == 1
    assert trainer.pair_counts[pair_space_a] == 1
    assert trainer.pair_counts[pair_ac] == 1

    trainer.train()

    assert trainer.merges[0] == (b"a", b"c")
