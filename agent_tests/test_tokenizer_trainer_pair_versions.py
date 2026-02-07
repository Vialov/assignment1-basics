from collections import defaultdict

from cs336_basics.bpe_tokenizer import TokenizerTrainer, pk


def test_merge_pair_updates_versions_for_merged_touched_and_new_pairs():
    trainer = TokenizerTrainer(
        file_path="unused.txt",
        dict_size=10,
        split_special_token="<|endoftext|>",
        special_tokens=[],
    )

    # One word "abc" with count=2.
    trainer.tokens = [b"a", b"b", b"c"]
    trainer.words = [b"abc"]
    trainer.word_counts = [2]
    trainer.word_heads = [0]
    trainer.token_list = [0, 1, 2]
    trainer.alive_tokens = [True, True, True]
    trainer.next_token = [1, 2, -1]
    trainer.prev_token = [-1, 0, 1]
    trainer.token_to_word = [0, 0, 0]

    pair_ab = pk(0, 1)
    pair_bc = pk(1, 2)
    pair_cc = pk(2, 2)  # untouched pair

    trainer.pair_counts = {
        pair_ab: 2,
        pair_bc: 2,
        pair_cc: 7,
    }
    trainer.pair_versions = {
        pair_ab: 4,
        pair_bc: 8,
        pair_cc: 11,
    }
    trainer.pair_occurrences = defaultdict(
        list,
        {
            pair_ab: [0],
            pair_bc: [1],
            pair_cc: [],
        },
    )
    trainer.pair_heap = []

    trainer.merge_pair(pair_ab)

    new_token_ind = len(trainer.tokens) - 1
    pair_new_c = pk(new_token_ind, 2)

    assert trainer.pair_versions[pair_ab] == 5
    assert trainer.pair_versions[pair_bc] == 9
    assert trainer.pair_versions[pair_new_c] == 1
    assert trainer.pair_versions[pair_cc] == 11

    assert trainer.pair_counts[pair_ab] == 0
    assert trainer.pair_counts[pair_bc] == 0
    assert trainer.pair_counts[pair_new_c] == 2
