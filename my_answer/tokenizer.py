from typing import Iterable, Iterator
import regex as re

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
        # TODO: resolve filepath
        f_vocab = open(vocab_filepath, "r")
        cls.vocab = f_vocab.read()
        f_vocab.close()

        f_merges = open(merges_filepath, "r")
        cls.merges = f_merges.read()
        f_merges.close()

        cls.special_tokens = special_tokens or []
        return

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

if __name__ == "__main__":
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
            6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

    t = Tokenizer(vocab, merges, ["<|endoftext|>"])
    print(t.encode("the<|endoftext|>cat<|endoftext|>ate"))
    print(t.decode([9, 7, 1, 5, 10, 3]))