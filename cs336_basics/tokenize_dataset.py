import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from cs336_basics.tokenizer import BPE


def _iter_documents(source, separator: str, chunk_size: int):
    """
    保留分隔符；按完整文档编码，避免在任意字符处切断 BPE 输入。
    chunk控制一次从磁盘读取多少
    """
    pending = ""
    while chunk := source.read(chunk_size):
        pending += chunk
        start = 0
        while (index := pending.find(separator, start)) != -1:
            end = index + len(separator)
            yield pending[start:end]
            start = end
        pending = pending[start:]
    if pending:
        yield pending


def tokenize_dataset(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    tokenizer_path: str | os.PathLike,
    document_separator: str = "<|endoftext|>",
    chunk_size: int = 1024 * 1024,
) -> int:
    """加载保存的 BPE,将 UTF-8 语料编码为可供 np.memmap 读取的 token ID 文件。

    input_path: 原始语料文本路径。
    output_path: 输出二进制文件路径（不是 .npy 文件，也不是目录）。
    tokenizer_path: train_BPE.py 保存的 (vocab, merges, special_tokens) 文件。
    token ID 固定以 uint32 保存，与训练脚本的读取类型一致。
    document_separator: 必须是已注册的特殊 token；语料中原有的分隔符会保留。
        没有分隔符时，整份文本作为一个文档处理，内存取决于最大文档长度。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.is_dir():
        raise ValueError("output_path must be a file path, for example data/train.bin")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    vocab, merges, special_tokens = torch.load(tokenizer_path, map_location="cpu", weights_only=True)
    tokenizer = BPE(vocab, merges, special_tokens)
    
    
    if not document_separator or document_separator not in tokenizer.special_tokens:
        raise ValueError("document_separator must be a nonempty registered special token")
    if any(document_separator in token and token != document_separator for token in tokenizer.special_tokens):
        raise ValueError("document_separator must not be contained in another special token")

    # 创建输出文件的父目录；允许创建多层目录，目录已存在时不报错。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0  # 累计已经写入的 token 数量。
    temporary_path = None  # 临时文件尚未创建，先用 None 占位。
    
    try:  # 尝试编码和保存；无论是否出错，最后都会进入 finally 清理。
        # 以 UTF-8 文本模式打开语料，newline="" 保留原始换行形式；退出 with 时自动关闭。
        with input_path.open("r", encoding="utf-8", newline="") as source:
            # 在输出目录创建临时二进制文件；文件名前缀为 .tokenize-，关闭后暂不自动删除。
            with NamedTemporaryFile(mode="wb", dir=output_path.parent, prefix=".tokenize-", delete=False) as target:
                temporary_path = Path(target.name)  # 记录临时文件路径，供后续替换或删除。
                # 逐篇获取文档，保留原有的文档分隔符，不一次性读取整份语料。
                for document in _iter_documents(source, document_separator, chunk_size):
                    token_ids = tokenizer.encode(document)  # 将当前文档转换成 token ID 列表。
                    # 转成 uint32 数组，再将原始二进制数据写入当前文件位置，不保存数组头信息。
                    np.asarray(token_ids, dtype=np.uint32).tofile(target)
                    total_tokens += len(token_ids)  # 把当前文档的 token 数加入总数。
        # 输入、输出文件都已关闭；编码全部成功后，将临时文件移到正式路径，替换已有目标文件。
        temporary_path.replace(output_path)
    finally:  # 正常结束或发生异常时，都执行下面的清理操作。
        if temporary_path is not None:  # 仅在临时文件已经创建并记录路径后尝试删除。
            # 失败时删除半成品；成功时临时文件已移走，missing_ok=True 允许文件不存在。
            temporary_path.unlink(missing_ok=True)

    print(f"Saved {total_tokens} tokens to {output_path} (uint32)")  # 显示 token 总数、保存路径和类型。
    return total_tokens


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Encode a text corpus using a saved BPE tokenizer")
    parser.add_argument("--input_path", required=True, help="UTF-8 text corpus")
    parser.add_argument("--output_path", required=True, help="Output binary token ID file")
    parser.add_argument("--tokenizer_path", required=True, help="BPE file saved by train_BPE.py")
    parser.add_argument("--document_separator", default="<|endoftext|>")
    parser.add_argument("--chunk_size", type=int, default=1024 * 1024, help="Number of text characters per read")
    return parser.parse_args(argv)


if __name__ == "__main__":
    tokenize_dataset(**vars(parse_args()))
