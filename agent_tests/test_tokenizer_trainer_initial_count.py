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


def test_initial_count_words_and_pairs(tmp_path):
    trainer = _run_initial_count(tmp_path, "hi hi")

    counts = dict(zip(trainer.words, trainer.word_counts))
    assert counts == {b"hi": 1, b" hi": 1}

    pair_hi = pk(ord("h"), ord("i"))
    pair_space_h = pk(ord(" "), ord("h"))
    assert trainer.pair_counts[pair_hi] == 2
    assert trainer.pair_counts[pair_space_h] == 1


def test_initial_count_token_links(tmp_path):
    trainer = _run_initial_count(tmp_path, "abc abc")

    # For each word, its token_list slice should match the word bytes,
    # and next/prev pointers should form a linked list for that word.
    for word_idx, word in enumerate(trainer.words):
        head = trainer.word_heads[word_idx]
        word_tokens = list(word)
        slice_end = head + len(word_tokens)

        assert trainer.token_list[head:slice_end] == word_tokens
        assert trainer.token_to_word[head:slice_end] == [word_idx] * len(word_tokens)

        if word_tokens:
            assert trainer.prev_token[head] == -1
            for offset in range(len(word_tokens) - 1):
                current = head + offset
                nxt = head + offset + 1
                assert trainer.next_token[current] == nxt
                assert trainer.prev_token[nxt] == current
            assert trainer.next_token[slice_end - 1] == -1
