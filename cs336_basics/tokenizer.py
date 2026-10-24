import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
import regex as re
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from tqdm import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPE():
    '''
    Byte Pair Encoding Tokenizer
    
    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.
    
    '''
    
    def __init__(self, 
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {value: key for key, value in vocab.items()} 
        
        # 防止vocab里面没有出现special_tokens,下面的操作把vocab里面的special_tokens补齐
        next_id = max(self.vocab, default=-1) + 1

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.reverse_vocab:
                self.vocab[next_id] = token_bytes
                self.reverse_vocab[token_bytes] = next_id
                next_id += 1   
    
    
    def _merged(self, tokens: list[bytes]) -> list[int]:
        for pair in self.merges:
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == pair
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # 下一条规则作用在本轮合并后的结果上
            tokens = new_tokens

        return [self.reverse_vocab[token] for token in tokens]


    def decode(self, token_ID):
        '''
        transfer the token ID into the corresponding token
        
        Args:
            token_ID (list[int]): The token ID list
                
        Returns:
            token_list: the token list corresponding to the token ID list
        '''
        token_list = []
        for s in token_ID:
            token_list.append(self.vocab[s])   
            
        decoded_text = b"".join(token_list).decode("utf-8", errors="replace")
        return decoded_text
        
    
    def encode(self, input: str):
        '''
        transfer the str into the corresponding token ID
        
        Args:
            token_ID (list[int]): The token ID list
        
        Returns:
            token_ID (list[int]): The token ID list
        
        '''
        token_ID = []    
        
        if self.special_tokens:
            escaped = [re.escape(s) 
                       for s in sorted(self.special_tokens, key=len, reverse=True)
                       ]
            # | 代表 or ,  () 代表以special_tokens划分的同时保留special_tokens
            pattern = "(" + "|".join(escaped) + ")"
            parts = re.split(pattern, input)
        else:
            parts = [input]           
        
        for part in parts:
            if part in self.special_tokens:
                token_ID.append(self.reverse_vocab[part.encode("utf-8")])
            elif part == "":
                continue
            else:
                # regex pre-tokenization
                pre_tokens = re.findall(PAT, part)
                
                for pre_token in pre_tokens:
                    # 每个 pre-token 转成 list[bytes]
                    pre_token_bytes = [bytes([b]) for b in pre_token.encode("utf-8")]
                    # BPE merge
                    part_token_ID = self._merged(pre_token_bytes)
                    token_ID.extend(part_token_ID)
                
        return token_ID

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

                
        
def _iter_training_parts(source, special_tokens: list[str], chunk_size: int):
    """分块读取，按特殊 token 边界产出正文；特殊 token 不参与训练统计。"""
    if not special_tokens:
        # 没有已知文档边界时，保留原来整体预分词的行为，避免任意切块改变结果。
        yield source.read()
        return
    if any(not token for token in special_tokens):
        raise ValueError("special_tokens must not contain empty strings")
    pattern = re.compile("|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)))
    longest_special = max(map(len, special_tokens))
    pending = ""
    while chunk := source.read(chunk_size):
        pending += chunk
        # 尾部可能只读到了较长特殊 token 的一部分，留到下一块再判断。
        safe_start = len(pending) - longest_special
        start = 0
        for match in pattern.finditer(pending):
            if match.start() > safe_start:
                break
            yield pending[start:match.start()]
            start = match.end()
        pending = pending[start:]
    # 文件结束后，剩余内容也按同样的规则切分，包括没有结束符的最后一篇。
    yield from pattern.split(pending)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 统计每种 pre-token 的出现次数
    pre_token_counts = Counter()


    chunk_size = kwargs.get("chunk_size", 1024 * 1024)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with open(input_path, "r", encoding="utf-8") as f:
        for part in _iter_training_parts(f, special_tokens, chunk_size):
            pre_tokens = re.findall(PAT, part)
            for pre_token in pre_tokens:
                pre_token_counts[pre_token] += 1
            
           
    ### step3: 初始化vocab,让他可以表示任何parts里面出现的划分后的字符串      
            
    # 1. 初始化全部 256 种单字节 token
    vocab = {i: bytes([i]) for i in range(256)}

    # 2. 加入特殊 token，它们各自占一个 ID
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # 3. 记录之后学到的合并规则
    merges = []
    
    ### step4: 把 pre-token 转成字节 tuple，保留出现次数
    word_counts = {}

    for pre_token, count in pre_token_counts.items():
        word = tuple(
            bytes([b]) for b in pre_token.encode("utf-8")
        )
        word_counts[word] = count

    ### step5: 反复统计和合并

    # pair -> 包含它的词；同一个词中出现多次，也只记录一次。
    # 此索引只用于缩小应用合并的范围，pair 频率仍在每轮全量重算。
    pair_words = defaultdict(set)
    for word in word_counts:
        for pair in zip(word, word[1:]):
            pair_words[pair].add(word)

    for _ in tqdm(
        range(max(0, vocab_size - len(vocab))),
        desc="BPE merges", unit="merge", dynamic_ncols=True, mininterval=1.0,
        disable=not kwargs.get("show_progress", True),
    ):
        # 每轮都重新统计，因为上一轮合并改变了相邻关系
        pair_counts = Counter()

        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        # 优先比较频率，频率相同则比较 pair 的字典序
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )

        # 记录新规则和新 token
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token

        # 原来这里遍历 word_counts.items()，重建全部 new_word_counts。
        # 此时 pair_words 中已经没有 best_pair 这个 key，但其他 pair 的集合里仍可能包含当前旧词。
        affected_words = pair_words.pop(best_pair)
        for word in affected_words:
            # count: word(由bytes组成) 在原本语料中的出现次数,这里只处理包含bestpair的word
            # 这里先把含有best pair的词从 word-count 表中移除
            count = word_counts.pop(word)
            
            # 移除旧词的索引
            for pair in set(zip(word, word[1:])):
                # 此时pair_words 里面已经没有含有best pair的词了
                # 所以indexed_words里面就是pair对应的word里面没有包含best pair的
                indexed_words = pair_words.get(pair)
                if indexed_words is not None:
                    # 从某个 pair 的集合中移除当前旧词
                    indexed_words.discard(word)
                    if not indexed_words:
                        del pair_words[pair]
                        
            new_word = []
            i = 0

            while i < len(word):
                if (
                    i + 1 < len(word)
                    and (word[i], word[i + 1]) == best_pair
                ):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word_counts[new_word] = count
            # 更新对应的pair_words记录，因为换表示了
            for pair in zip(new_word, new_word[1:]):
                pair_words[pair].add(new_word)

    return vocab, merges
