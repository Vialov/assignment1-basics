from cs336_basics.bpe_tokenizer import TokenizerTrainer


def test_initial_count_excludes_special_tokens_from_words(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello <|endoftext|> world", encoding="utf-8")

    trainer = TokenizerTrainer(
        file_path=str(file_path),
        dict_size=256,
        split_special_token="<|endoftext|>",
        special_tokens=["<|endoftext|>"],
    )
    trainer.initial_count(num_chunks=1, max_workers=1)

    assert b"<|endoftext|>" not in trainer.words
