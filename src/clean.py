"""法规文本清洗：去页码、页眉页脚、空行、不可见字符，统一标点。

用法：python -m src.clean
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

from src.config import RAW_CLEAN_DIR, RAW_DIR, configure_logging

logger = logging.getLogger(__name__)

# 页码行：- 12 -、12、— 12 —、第 12 页、第 12 页 共 20 页
PAGE_PATTERNS = [
    re.compile(r"^[\s\-—–_]*\d{1,4}[\s\-—–_]*$"),
    re.compile(r"^第\s*\d{1,4}\s*页(\s*[/／共]?\s*\d{1,4}\s*页?)?$"),
    re.compile(r"^共\s*\d{1,4}\s*页$"),
    re.compile(r"^[\s\-—–_]*页码[:：]?\s*\d{1,4}[\s\-—–_]*$"),
]
# 页眉页脚常见噪声：网址、文档来源标记
NOISE_PATTERNS = [
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"^(资料来源|来源|编辑|排版)[:：].*$"),
]
INVISIBLE_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060\u00ad\u00a0"), None)
CJK = r"\u4e00-\u9fff"
HALF_TO_FULL = {",": "，", ";": "；", ":": "：", "?": "？", "!": "！", "(": "（", ")": "）"}


def normalize_punct(text: str) -> str:
    """统一标点为全角、清除不可见字符。

    Args:
        text: 原始文本。

    Returns:
        标点规范化后的文本。
    """
    cleaned = text.translate(INVISIBLE_CHARS).replace("\u3000", " ")
    chars: List[str] = []
    for index, char in enumerate(cleaned):
        prev = cleaned[index - 1] if index else ""
        if char in HALF_TO_FULL and re.match(f"[{CJK}]", prev or " "):
            chars.append(HALF_TO_FULL[char])
        else:
            chars.append(char)
    result = "".join(chars)
    result = re.sub(rf"(?<=[{CJK}])\.(?=[{CJK}])", "。", result)
    return result


def clean_text(raw: str) -> str:
    """清洗单部法规全文。

    Args:
        raw: 文件原始内容。

    Returns:
        清洗后的文本，一行一段，条文编号与章标题原样保留。
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    kept: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip().strip("\u3000").strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in PAGE_PATTERNS):
            continue
        if any(pattern.search(stripped) for pattern in NOISE_PATTERNS):
            continue
        kept.append(stripped)
    merged = "\n".join(kept)
    merged = re.sub(r"\n{2,}", "\n", merged)
    return normalize_punct(merged).strip()


def clean_file(src: Path, out_dir: Path = RAW_CLEAN_DIR) -> Path:
    """清洗单个 txt 并写入 raw_clean 目录（同名覆盖）。

    Args:
        src: 原始法规 txt 路径。
        out_dir: 输出目录。

    Returns:
        清洗结果文件路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = src.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    out_path = out_dir / src.name
    out_path.write_text(cleaned, encoding="utf-8")
    logger.info("清洗完成 %s：%d 字 -> %d 字", src.name, len(raw), len(cleaned))
    return out_path


def count_articles(text: str) -> int:
    """统计文本中"第X条"出现次数。"""
    return len(re.findall(r"第[一二三四五六七八九十百零]+条", text))


def main() -> None:
    """遍历 data/raw 下全部 txt，清洗到 data/raw_clean 并统计条数。"""
    configure_logging()
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        logger.warning("data/raw 为空，请先放入法规 txt 文件")
        return
    stats: Dict[str, int] = {}
    for path in files:
        out_path = clean_file(path)
        stats[path.stem] = count_articles(out_path.read_text(encoding="utf-8"))
    print(f"已清洗 {len(stats)} 部法规 -> {RAW_CLEAN_DIR}")
    for name, count in stats.items():
        print(f"  {name}: 第X条 出现 {count} 次")
    logger.info("清洗汇总：%d 部法规，共 %d 条条文", len(stats), sum(stats.values()))


if __name__ == "__main__":
    main()
