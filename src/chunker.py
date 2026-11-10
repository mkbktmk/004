"""条文切块：按"第X条"切分，过长条文按款二次切分。

用法：python -m src.chunker
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from src.config import CHUNK_MAX_LEN, CHUNKS_PATH, RAW_CLEAN_DIR, RAW_DIR, configure_logging
from src.clean import clean_text

logger = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零]+条")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零]+章\s*\S*.*$", re.MULTILINE)
ARTICLE_NUM_RE = re.compile(r"第([一二三四五六七八九十百零]+)条")

SHORT_NAME_MAP: Dict[str, str] = {
    "中华人民共和国劳动合同法": "劳动合同法",
    "中华人民共和国劳动法": "劳动法",
    "中华人民共和国劳动争议调解仲裁法": "劳动争议调解仲裁法",
    "中华人民共和国社会保险法": "社会保险法",
    "职工带薪年休假条例": "职工带薪年休假条例",
    "工伤保险条例": "工伤保险条例",
}


def short_law_name(stem: str) -> str:
    """由文件名推导法规简称。

    Args:
        stem: 文件主名，如"中华人民共和国劳动合同法"。

    Returns:
        法规简称，如"劳动合同法"。
    """
    if stem in SHORT_NAME_MAP:
        return SHORT_NAME_MAP[stem]
    return stem.replace("中华人民共和国", "").strip() or stem


def _split_long(law: str, article_no: str, chapter: str, text: str) -> List[Dict]:
    """超长条文按款二次切分，共享同一 article_no。

    Args:
        law: 法规简称。
        article_no: 条号原文，如"第四十七条"。
        chapter: 章名，可为空字符串。
        text: 条文正文。

    Returns:
        切块列表。
    """
    if len(text) <= CHUNK_MAX_LEN:
        return [
            {"id": f"{law}-{article_no}", "law": law, "article_no": article_no,
             "chapter": chapter, "text": text, "part": 1}
        ]
    pieces: List[str] = []
    buffer = ""
    for line in text.split("\n"):
        while len(line) > CHUNK_MAX_LEN:  # 单款本身超长时硬切
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(line[:CHUNK_MAX_LEN])
            line = line[CHUNK_MAX_LEN:]
        if not line:
            continue
        if len(buffer) + len(line) + 1 <= CHUNK_MAX_LEN:
            buffer = f"{buffer}\n{line}" if buffer else line
        else:
            if buffer:
                pieces.append(buffer)
            buffer = line
    if buffer:
        pieces.append(buffer)
    return [
        {"id": f"{law}-{article_no}-{index}", "law": law, "article_no": article_no,
         "chapter": chapter, "text": piece.strip(), "part": index}
        for index, piece in enumerate(pieces, start=1)
    ]


def split_law(law: str, text: str) -> List[Dict]:
    """将单部法规全文切分为条文块。

    Args:
        law: 法规简称。
        text: 清洗后的法规全文。

    Returns:
        条文块列表，每块含 id/law/article_no/chapter/text/part。
    """
    chapters = [(m.start(), m.group(0).strip()) for m in CHAPTER_RE.finditer(text)]
    matches = list(ARTICLE_RE.finditer(text))
    # 只把"行首"的条号视为条文起点，避免把正文里"依照本法第三十八条"这类引用切成分块
    line_start = [m for m in matches if m.start() == 0 or text[m.start() - 1] == "\n"]
    if len(line_start) >= 2:
        if len(line_start) != len(matches):
            logger.info("%s：忽略 %d 处正文内条号引用", law, len(matches) - len(line_start))
        matches = line_start
    else:
        logger.warning("%s 条文未按行分列，回退为全文匹配", law)
    chunks: List[Dict] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_no = match.group(0)
        chapter = ""
        for position, title in chapters:
            if position < start:
                chapter = title
            else:
                break
        chunks.extend(_split_long(law, article_no, chapter, text[start:end].strip()))
    logger.info("切分 %s：%d 条 -> %d 块", law, len(matches), len(chunks))
    return chunks


def build_chunks() -> List[Dict]:
    """读取法规文本，产出 data/chunks.jsonl。

    Returns:
        全部切块列表。
    """
    source_dir = RAW_CLEAN_DIR if any(RAW_CLEAN_DIR.glob("*.txt")) else RAW_DIR
    files = sorted(source_dir.glob("*.txt"))
    if not files:
        logger.error("没有可切分的法规文本，请先执行 python -m src.clean 或放入 data/raw/*.txt")
        return []
    if source_dir is RAW_DIR:
        logger.warning("data/raw_clean 为空，直接使用 data/raw 并即时清洗")
    all_chunks: List[Dict] = []
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = raw if source_dir is RAW_CLEAN_DIR else clean_text(raw)
        all_chunks.extend(split_law(short_law_name(path.stem), text))
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("切块完成：%d 块 -> %s", len(all_chunks), CHUNKS_PATH)
    return all_chunks


def report_stats(chunks: List[Dict]) -> None:
    """打印块数、平均长度、最长块所属条文。"""
    if not chunks:
        print("没有切块数据")
        return
    lengths = [len(chunk["text"]) for chunk in chunks]
    longest = max(chunks, key=lambda chunk: len(chunk["text"]))
    print(f"总块数：{len(chunks)}")
    print(f"平均长度：{sum(lengths) / len(lengths):.1f} 字")
    print(f"最长块：{longest['law']}{longest['article_no']}（{len(longest['text'])} 字）")


def main() -> None:
    """命令行入口：切块并打印统计。"""
    configure_logging()
    report_stats(build_chunks())


if __name__ == "__main__":
    main()
