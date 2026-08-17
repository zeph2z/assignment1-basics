import time
from tqdm import tqdm

def train_bpe(input_path, vocab_size, special_tokens):
    start = time.time()

    f = open(input_path, "r")
    text = f.read()
    f.close()

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    counts = {}

    # special tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # pretokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    import regex as re
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)

    for chunk in tqdm(chunks, desc="pretokenize", unit="chunk"):
        for it in re.finditer(PAT, chunk):
            word = it.group()
            key = tuple(bytes([i]) for i in word.encode("utf-8"))
            counts[key] = counts.get(key, 0) + 1

    # training
    pbar = tqdm(total=vocab_size - len(vocab), desc="merges", unit="merge")
    while len(vocab) < vocab_size:
        pair_counts = {}

        for key, count in counts.items():
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_vocab_entry = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_vocab_entry

        new_counts = {}

        for key, count in counts.items():
            new_key = []
            i = 0
            while i < len(key) - 1:
                if key[i] == best_pair[0] and key[i + 1] == best_pair[1]:
                    new_key.append(new_vocab_entry)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            if i == len(key) - 1:
                new_key.append(key[i])
            new_counts[tuple(new_key)] = counts[key]

        counts = new_counts
        pbar.update(1)

    pbar.close()
    elasped = time.time() - start
    print(f"Training costs {elasped:.2f} seconds.")

    return vocab, merges

if __name__ == "__main__":
    input_path = "tests/fixtures/corpus.en"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("vocab size:", len(vocab))
    print("merges size:", len(merges))
    for i in sorted(vocab)[:10]:
        print(i, vocab[i])
