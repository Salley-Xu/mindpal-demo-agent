"""知识提取管线 — PDF → 文本 → LLM 结构化 → KnowledgeDocument JSON

三阶段管线：
  Stage 1: PDF 文本提取     — pdfminer.six → 纯文本 + 缓存到 .txt
  Stage 2: LLM 结构化提取   — DeepSeek Chat → KnowledgeDocument JSON
  Stage 3: 入库 + 去重      — 追加写入 knowledge_corpus/

用法：
  # 处理所有 knowledge_source/ 下的 PDF
  python knowledge_ingest.py --all

  # 处理单个 PDF
  python knowledge_ingest.py --pdf who_mhgap_ig.pdf

  # 只提取文本不结构化
  python knowledge_ingest.py --pdf who_mhgap_ig.pdf --extract-only

  # 从已有的 .txt 文件结构化
  python knowledge_ingest.py --txt knowledge_source/who_mhgap_ig.txt

  # Dry-run（不写文件）
  python knowledge_ingest.py --all --dry-run
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("knowledge_ingest")

# ── 路径 ──

BACKEND_DIR = Path(__file__).resolve().parent
KNOWLEDGE_SOURCE_DIR = BACKEND_DIR / "data" / "knowledge_source"
KNOWLEDGE_CORPUS_DIR = BACKEND_DIR / "data" / "knowledge_corpus"

# ── 配置 ──

# 导入 config（复用项目原有的 API 配置）
sys.path.insert(0, str(BACKEND_DIR))
from config import config as app_config

# LLM 结构化 Prompt
EXTRACT_SYSTEM_PROMPT = """你是一位心理学知识管理专家。你的任务是从给定的心理学文献片段中提取结构化的专业知识条目。

每条知识条目必须严格遵循以下 JSON Schema：

{
  "id": "字符串，格式为 kb_{category}_{序号}，序号用自增数字",
  "title": "知识标题，中文为主，英文术语保留原文，20字以内",
  "summary": "一句话摘要，约100字，概括核心内容",
  "content": "详细内容，用Markdown格式。包含核心概念、操作方法、步骤、注意事项等。300-800字。保留原文中的具体数据、步骤编号、关键引用。",
  "source_reference": "来源引用，格式：作者/机构 (年份). 文献标题. 出版社/期刊。优先保留原文中的引用信息",
  "category": "知识分类，从以下列表中选择最匹配的: cbt_core | cbt_technique | crisis_intervention | dbt_emotion | grounding | psych_concept | cultural_adaptation | clinical_assessment | medication_knowledge | other",
  "tags": ["标签列表，5-10个中英文关键词，用于检索"],
  "related_risk_levels": ["适用风险等级: level_0 | level_1 | level_2 | level_3"],
  "applicability": "适用情境描述，1-2句话",
  "contraindications": "禁忌或不适用场景，1句话",
  "dialogue_example": "如果适用，给一段简短的Agent对话示例；如果不适用，留空字符串",
  "language": "zh"
}

提取规则：
1. 每个独立的知识概念或干预技术提取为一条独立条目，不要合并。
2. 保留原文中的关键数据、步骤编号、术语和引用信息。
3. 如果原文片段没有完整的心理学知识内容（如仅是目录、致谢、引用列表），返回空数组 []。
4. category 优先选择最匹配的：精神障碍诊断/药物 → clinical_assessment，干预技术 → cbt_technique 或 dbt_emotion，危机管理 → crisis_intervention。
5. tags 要包含中英文关键词，便于检索。例如：["认知重构", "cognitive restructuring", "CBT", "自动思维"]。
6. dialogue_example 只在内容适合用于对话指导时才提供，否则留空。
7. 不要编造信息——只提取原文中确实存在的内容。
8. 输出必须是合法 JSON 数组，不含任何额外文字。

示例输出格式：
[{"id": "kb_mhgap_001", "title": "...", "summary": "...", "content": "...", "source_reference": "...", "category": "...", "tags": [...], "related_risk_levels": [...], "applicability": "...", "contraindications": "...", "dialogue_example": "...", "language": "zh"}]"""


# ── 数据模型 ──

class ExtractedEntry(BaseModel):
    """LLM 提取的知识条目（id 字段由管线自动重写）"""
    id: str = ""
    title: str
    summary: str = ""
    content: str
    source_reference: str = ""
    category: str = "other"
    tags: List[str] = []
    related_risk_levels: List[str] = []
    applicability: str = ""
    contraindications: str = ""
    dialogue_example: str = ""
    language: str = "zh"


# ── Stage 1: PDF 提取 ──

def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """提取 PDF 纯文本，缓存到同名 .txt"""
    txt_path = pdf_path.with_suffix(".txt")

    # 优先读缓存
    if txt_path.exists():
        cached = txt_path.read_text(encoding="utf-8")
        if len(cached) > 500:
            logger.info(f"使用缓存: {txt_path.name} ({len(cached)} 字符)")
            return cached

    try:
        from pdfminer.high_level import extract_text
        logger.info(f"提取 PDF: {pdf_path.name}")
        text = extract_text(str(pdf_path))
        txt_path.write_text(text, encoding="utf-8")
        logger.info(f"已保存: {txt_path.name} ({len(text)} 字符)")
        return text
    except ImportError:
        logger.error("需要 pdfminer.six: pip install pdfminer.six")
        return None
    except Exception as e:
        logger.error(f"PDF 提取失败 {pdf_path.name}: {e}")
        return None


def chunk_text_for_extraction(text: str, max_chars: int = 2500) -> List[str]:
    """将长文本切分为适合 LLM 处理的块，在段落边界处切分"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 跳过明显的噪音行（页码、页眉等）
        if re.match(r'^[\d\sivxlcdm]+$', para, re.IGNORECASE):
            continue
        if len(para) < 20:  # 太短的可能是页码/章节号
            if current:
                current += "\n" + para
            continue

        if len(current) + len(para) < max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current and len(current) > 200:
                chunks.append(current)
            current = para
    if current and len(current) > 200:
        chunks.append(current)
    return chunks


# ── Stage 2: LLM 结构化提取 ──

class KnowledgeExtractor:
    """用 DeepSeek Chat 将文本片段结构化为 KnowledgeDocument"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=app_config.DEEPSEEK_API_KEY,
            base_url=app_config.API_BASE_URL,
        )
        self.model = app_config.CHAT_MODEL
        self.id_prefix = ""         # 每个 PDF 重置
        self.entry_counter = 0      # 全局自增
        self.seen_hashes: set = set()  # 内容去重

    async def extract_chunk(
        self,
        text: str,
        attempt: int = 1,
    ) -> List[Dict]:
        """从一个文本块中提取知识条目（带重试）"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"请从以下文本中提取结构化的心理学知识条目：\n\n{text}"},
                ],
                temperature=0.15,   # 低温度保证一致性
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw)
        except Exception as e:
            logger.warning(f"提取失败 (attempt={attempt}): {e}")
            if attempt < 3:
                await asyncio.sleep(2)
                return await self.extract_chunk(text, attempt + 1)
            return []

    def _parse_response(self, raw: str) -> List[Dict]:
        """解析 LLM 响应，提取 JSON 数组"""
        # 移除可能的 markdown 代码块包装
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON 数组
            match = re.search(r'\[\s*\{.*\}\s*\]', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"JSON 解析失败: {raw[:200]}...")
                    return []
            else:
                logger.warning(f"未找到 JSON 数组: {raw[:200]}...")
                return []

        if isinstance(data, dict):
            # 有些模型可能返回 {"entries": [...]}
            for key in ["entries", "items", "documents", "results"]:
                if key in data:
                    data = data[key]
                    break
            else:
                data = [data] if data else []

        if not isinstance(data, list):
            return []
        return data

    def deduplicate(self, entries: List[Dict]) -> List[Dict]:
        """基于标题 + 内容哈希去重（跨PDF全局 + 与已有语料对比）"""
        result = []
        for entry in entries:
            title = entry.get("title", "")
            content = entry.get("content", "")
            h = hashlib.md5(f"{title}|{content[:200]}".encode()).hexdigest()
            if h in self.seen_hashes:
                logger.debug(f"跳过重复: {title}")
                continue
            self.seen_hashes.add(h)
            result.append(entry)
        return result

    def finalize_entry(self, entry: Dict, source_file: str, idx: int) -> Dict:
        """重写 id，添加来源信息"""
        category = entry.get("category", "other") or "other"
        content_hash = hashlib.md5(entry.get("content", "").encode()).hexdigest()[:8]
        entry["id"] = f"kb_{category}_{content_hash}"
        entry["source_file"] = source_file
        entry["created_at"] = datetime.now().isoformat()
        # 确保必要字段存在
        entry.setdefault("summary", entry.get("content", "")[:100])
        entry.setdefault("tags", [])
        entry.setdefault("related_risk_levels", ["level_0", "level_1"])
        entry.setdefault("applicability", "")
        entry.setdefault("contraindications", "")
        entry.setdefault("dialogue_example", "")
        entry.setdefault("language", "zh")
        return entry

    async def process_text(
        self,
        text: str,
        source_file: str,
        chunk_size: int = 2500,
        max_parallel: int = 3,
    ) -> List[Dict]:
        """处理一个 PDF 的完整文本 → 结构化条目列表"""
        chunks = chunk_text_for_extraction(text, max_chars=chunk_size)
        logger.info(f"{source_file}: {len(chunks)} 个文本块待处理")

        if not chunks:
            return []

        # 并行处理文本块
        all_entries = []
        sem = asyncio.Semaphore(max_parallel)

        async def process_one(chunk: str, idx: int):
            async with sem:
                logger.info(f"  [{idx+1}/{len(chunks)}] 处理中...")
                entries = await self.extract_chunk(chunk)
                if entries:
                    logger.info(f"  [{idx+1}/{len(chunks)}] 提取到 {len(entries)} 条")
                return entries

        tasks = [process_one(c, i) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        for entries in results:
            all_entries.extend(entries)

        # 去重 + 编 id
        all_entries = self.deduplicate(all_entries)
        for i, entry in enumerate(all_entries):
            all_entries[i] = self.finalize_entry(entry, source_file, i)

        logger.info(f"{source_file}: 共提取 {len(all_entries)} 条有效知识条目（去重后）")
        return all_entries


# ── Stub: JSON 行格式化，避免 Windows cp1252 编码问题 ──

class _JSONLinesStub:
    @staticmethod
    def dumps(obj, **kw):
        return json.dumps(obj, ensure_ascii=False, **kw)


# ── Stage 3: 入库 ──

def load_existing_ids(corpus_dir: Path = KNOWLEDGE_CORPUS_DIR) -> set:
    """加载已有知识 ID（用于全局去重）"""
    existing = set()
    if not corpus_dir.is_dir():
        return existing
    for fname in corpus_dir.glob("*.json"):
        try:
            data = json.loads(fname.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else [data]
            for item in items:
                existing.add(item.get("id", ""))
                # 也记录标题哈希用于模糊去重
                title = item.get("title", "")
                content = item.get("content", "")[:200]
                h = hashlib.md5(f"{title}|{content}".encode()).hexdigest()
                existing.add(f"hash:{h}")
        except Exception:
            pass
    return existing


def save_entries(
    entries: List[Dict],
    source_file: str,
    corpus_dir: Path = KNOWLEDGE_CORPUS_DIR,
    dry_run: bool = False,
) -> str:
    """将条目保存为知识语料 JSON 文件"""
    if not entries:
        return ""

    # 按来源文件命名输出文件
    stem = Path(source_file).stem
    out_name = f"kb_{stem}.json"
    out_path = corpus_dir / out_name

    # 合并已有条目（如果文件已存在）
    existing = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 去重
    existing_ids = {e["id"] for e in existing}
    new_entries = [e for e in entries if e["id"] not in existing_ids]
    if not new_entries:
        logger.info(f"{out_name}: 无新增条目（全部已存在）")
        return ""

    all_entries = existing + new_entries

    if dry_run:
        logger.info(f"[DRY-RUN] 将写入 {out_name}: {len(new_entries)} 条新增")
        for e in new_entries[:3]:
            logger.info(f"  - {e['title']}")
        return ""

    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"已写入: {out_name} ({len(all_entries)} 条, +{len(new_entries)} 新增)")
    return str(out_path)


# ── 主流程 ──

async def process_pdf(
    pdf_path: str,
    extractor: KnowledgeExtractor,
    dry_run: bool = False,
    extract_only: bool = False,
) -> int:
    """处理单个 PDF 的完整管线"""
    pdf = Path(pdf_path)
    if not pdf.exists():
        # 尝试在 knowledge_source 目录下查找
        pdf = KNOWLEDGE_SOURCE_DIR / pdf.name

    if not pdf.exists():
        logger.error(f"PDF 不存在: {pdf_path} (也尝试了 {pdf})")
        return 0

    # Stage 1: 提取文本
    text = extract_pdf_text(pdf)
    if not text:
        return 0

    if extract_only:
        logger.info(f"文本提取完成 ({len(text)} 字符)，跳过结构化")
        return 0

    # Stage 2: LLM 结构化
    if dry_run:
        # Dry-run: 展示 chunk 信息但不调 LLM
        chunks = chunk_text_for_extraction(text, 2500)
        logger.info(f"[DRY-RUN] {pdf.name}: {len(chunks)} 个文本块，跳过 LLM 调用")
        logger.info(f"[DRY-RUN] 预估输入 ~{sum(len(c)//2 for c in chunks):,} tokens")
        return 0

    entries = await extractor.process_text(text, source_file=pdf.name)

    # Stage 3: 入库
    if entries:
        save_entries(entries, pdf.name, dry_run=dry_run)

    return len(entries)


async def process_all(
    source_dir: Path = KNOWLEDGE_SOURCE_DIR,
    dry_run: bool = False,
    extract_only: bool = False,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """处理 knowledge_source/ 下所有 PDF"""
    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        logger.info("没有找到 PDF 文件")
        return {}

    extractor = KnowledgeExtractor()

    # 预加载已有 ID 去重
    existing_ids = load_existing_ids()
    extractor.seen_hashes = {
        eid.replace("hash:", "") for eid in existing_ids if eid.startswith("hash:")
    }
    logger.info(f"已有 {len(existing_ids)} 个已知 ID（含哈希）")

    results = {}
    for pdf in pdfs:
        # 跳过已有对应 JSON 的
        stem = pdf.stem
        out_file = KNOWLEDGE_CORPUS_DIR / f"kb_{stem}.json"
        if skip_existing and out_file.exists():
            logger.info(f"跳过（已有输出）: {pdf.name}")
            continue

        n = await process_pdf(str(pdf), extractor, dry_run, extract_only)
        results[pdf.name] = n

    return results


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="知识提取管线：PDF → 结构化知识条目")
    parser.add_argument("--all", action="store_true", help="处理所有 PDF")
    parser.add_argument("--pdf", type=str, help="处理单个 PDF 文件")
    parser.add_argument("--txt", type=str, help="从已有 .txt 文件结构化")
    parser.add_argument("--dry-run", action="store_true", help="不写文件")
    parser.add_argument("--extract-only", action="store_true", help="只提取文本不结构化")
    parser.add_argument("--chunk-size", type=int, default=2500, help="LLM 文本块大小")
    parser.add_argument("--max-parallel", type=int, default=3, help="并行处理数")
    args = parser.parse_args()

    async def run():
        if args.txt:
            # 从已有 txt 结构化
            txt_path = Path(args.txt)
            if not txt_path.exists():
                txt_path = KNOWLEDGE_SOURCE_DIR / txt_path.name
            if not txt_path.exists():
                logger.error(f"TXT 不存在: {args.txt}")
                return
            text = txt_path.read_text(encoding="utf-8")
            extractor = KnowledgeExtractor()
            existing_ids = load_existing_ids()
            extractor.seen_hashes = {
                eid.replace("hash:", "") for eid in existing_ids if eid.startswith("hash:")
            }
            entries = await extractor.process_text(
                text, source_file=txt_path.with_suffix(".pdf").name
            )
            if entries:
                save_entries(entries, txt_path.with_suffix(".pdf").name, dry_run=args.dry_run)
        elif args.all:
            results = await process_all(dry_run=args.dry_run, extract_only=args.extract_only)
            total = sum(results.values())
            logger.info(f"处理完成: {len(results)} 个 PDF, {total} 条新知识")
        elif args.pdf:
            extractor = KnowledgeExtractor()
            existing_ids = load_existing_ids()
            extractor.seen_hashes = {
                eid.replace("hash:", "") for eid in existing_ids if eid.startswith("hash:")
            }
            n = await process_pdf(
                args.pdf, extractor,
                dry_run=args.dry_run, extract_only=args.extract_only,
            )
            logger.info(f"处理完成: {n} 条新知识")
        else:
            parser.print_help()

    asyncio.run(run())


if __name__ == "__main__":
    main()
