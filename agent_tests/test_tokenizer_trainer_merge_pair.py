import pytest

from cs336_basics.bpe_tokenizer import TokenizerTrainer, pk


def _run_initial_count(tmp_path, text: str) -> TokenizerTrainer:
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text, encoding="utf-8")
    trainer = TokenizerTrainer(
        file_path=str(file_path),
        dict_size=256,
        split_special_token="<|endoftext|>",
        special_tokens=None,
    )
    trainer.initial_count(num_chunks=1, max_workers=1)
    return trainer


def test_merge_pair_updates_links_and_counts(tmp_path):
    trainer = _run_initial_count(tmp_path, "aba")

    pair_ab = pk(ord("a"), ord("b"))
    pair_ba = pk(ord("b"), ord("a"))

    trainer.merge_pair(pair_ab)

    new_token_ind = len(trainer.tokens) - 1
    assert trainer.tokens[new_token_ind] == b"ab"
    assert trainer.merges[-1] == (b"a", b"b")

    # Token list should now be ["ab", "b"(dead), "a"] for the single word
    assert trainer.token_list[0] == new_token_ind
    assert trainer.alive_tokens[1] is False
    assert trainer.token_list[2] == ord("a")

    assert trainer.prev_token[0] == -1
    assert trainer.next_token[0] == 2
    assert trainer.prev_token[2] == 0
    assert trainer.next_token[2] == -1

    assert trainer.pair_counts[pair_ab] == 0
    assert trainer.pair_counts[pair_ba] == 0

    pair_new_a = pk(new_token_ind, ord("a"))
    assert trainer.pair_counts[pair_new_a] == 1
    assert trainer.pair_occurrences[pair_new_a] == [0]
    assert trainer.pair_occurrences[pair_ab] == []


def test_merge_pair_multiple_occurrences(tmp_path):
    trainer = _run_initial_count(tmp_path, "hi hi")

    pair_hi = pk(ord("h"), ord("i"))
    pair_space_h = pk(ord(" "), ord("h"))

    trainer.merge_pair(pair_hi)

    new_token_ind = len(trainer.tokens) - 1
    assert trainer.tokens[new_token_ind] == b"hi"

    assert trainer.pair_counts[pair_hi] == 0
    assert trainer.pair_counts[pair_space_h] == 0

    pair_space_hi = pk(ord(" "), new_token_ind)
    assert trainer.pair_counts[pair_space_hi] == 1

    # Two merged occurrences: "hi" and " hi"
    assert trainer.alive_tokens[1] is False
    assert trainer.alive_tokens[4] is False
