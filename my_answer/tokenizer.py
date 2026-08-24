from typing import Iterable, Iterator
import regex as re
import json, os
from tqdm import tqdm
from my_answer.bpe import train_bpe
import numpy as np
class Tokenizer:

    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] = None):
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=len, reverse=True)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab_reverse : dict[bytes, int] = {vocab[i]: i for i in vocab.keys()}
        self.merge_rank = {pair: idx for idx, pair in enumerate(merges)}

        if special_tokens is not None:
            for special_token in special_tokens:
                st = special_token.encode("utf-8")
                if st not in self.vocab_reverse:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = st
                    self.vocab_reverse[st] = new_id

        return

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)

        vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        output = []
        
        if self.special_tokens:
            chunks = re.split("(" + "|".join(map(re.escape, self.special_tokens)) + ")", text)
        else:
            chunks = [text]

        for chunk in chunks:
            if chunk in self.special_tokens:
                output.append(self.vocab_reverse[chunk.encode("utf-8")])
                continue

            for it in re.finditer(self.PAT, chunk):
                word = it.group()
                key = list(bytes([i]) for i in word.encode("utf-8"))

                while True:
                    best_i = None
                    best_rank = None

                    for i in range(len(key) - 1):
                        r = self.merge_rank.get((key[i], key[i + 1]))
                        if r is not None and (best_rank is None or r < best_rank):
                            best_i, best_rank = i, r

                    if best_i is None:
                        break

                    key[best_i] = key[best_i] + key[best_i + 1]
                    del key[best_i + 1]

                output.extend(self.vocab_reverse[token] for token in key)

        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        carry = ""
        TAIL = 4096

        for chunk in iterable:
            text = carry + chunk
            limit = len(text) - TAIL

            cut = 0
            for m in re.finditer(self.PAT, text):
                if m.end() > limit:
                    break
                cut = m.end()

            if cut > 0:
                yield from self.encode(text[:cut])
                carry = text[cut:]
            else:
                carry = text

        if carry:
            yield from self.encode(carry)

    def decode(self, ids: list[int]) -> str:
        output = b"".join(self.vocab[id] for id in ids)
        return output.decode("utf-8", errors="replace")

    def save_tokenizer(self, vocab_filepath, merges_filepath):
        json_vocab = {str(k): v.decode("latin-1") for k, v in self.vocab.items()}
        json_merges = [[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges]

        with open(vocab_filepath, "w") as f:
            json.dump(json_vocab, f, indent=2)
        with open(merges_filepath, "w") as f:
            json.dump(json_merges, f, indent=2)

def train(use_cache = False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "..", "data", "TinyStoriesV2-GPT4-train.txt")    
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab_filepath = os.path.join(BASE_DIR, "..", "data", "TinyStoriesVocab.json")
    merges_filepath = os.path.join(BASE_DIR, "..", "data", "TinyStoriesMerges.json")

    if use_cache:
        tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    else:
        vocab, merges = train_bpe(path, vocab_size, special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        tokenizer.save_tokenizer(vocab_filepath, merges_filepath)

    with open(path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in tqdm(f, desc="counting lines", unit="line"))

    total_bytes = os.path.getsize(path)

    encode_iterator = None

    with open(path, "r", encoding="utf-8") as f:
        lines = tqdm(f, total=num_lines, desc="encoding", unit="line")
        encode_iterator = tokenizer.encode_iterable(lines)

        arr = np.fromiter(encode_iterator, dtype=np.uint16)
        arr_path = os.path.join(BASE_DIR, "..", "data", "TinyStoriesEncodeArray.bin")
        np.save(arr_path, arr)    

    print(f"{total_bytes / arr.size:.3f} bytes/token")

if __name__ == "__main__":
    train(True)