# -*- coding: utf-8 -*-
"""
Agent Benchmark 数据集校验器（Phase 0.5 Task 0.5.4）。

用法：
  python -m evaluation.validate_dataset evaluation/datasets/agent_benchmark_v1_1.jsonl

检查项（对齐 Phase 0.5 §20 Dataset Validation）：
  - Schema 合法
  - case_id 唯一
  - conversation 非空、角色合法
  - 标签合法（intent / emotion / risk / primary / tool / safety）
  - tool_actions 去重
  - precondition 合法
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from evaluation.benchmark_schema import BenchmarkCase, EMOTION_LABELS, IntentLabel


def validate_file(path: Path, verbose: bool = True) -> dict:
    errors = []
    warnings = []
    ids = set()
    n = 0
    tool_dup = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: JSON 解析失败 - {e}")
                continue
            try:
                case = BenchmarkCase.model_validate(obj)
            except ValidationError as e:
                errors.append(f"line {line_no} ({obj.get('case_id', '?')}): Schema 校验失败 - {str(e)[:200]}")
                continue

            # case_id 唯一
            cid = case.case_id
            if cid in ids:
                errors.append(f"case_id 重复: {cid}")
            ids.add(cid)

            # conversation 非空 + 角色合法
            if not case.conversation:
                errors.append(f"{cid}: conversation 为空")
            for turn in case.conversation:
                if turn.role.value not in {"user", "assistant"}:
                    errors.append(f"{cid}: 非法角色 {turn.role.value}")
            if not any(t.role.value == "user" for t in case.conversation):
                errors.append(f"{cid}: 没有 user 轮次")

            # emotion 合法
            exp = case.expected
            if exp.emotion is not None and exp.emotion not in EMOTION_LABELS:
                errors.append(f"{cid}: 非法 emotion '{exp.emotion}'")

            # tool_actions 去重
            seen = set()
            for t in exp.tool_actions:
                if t.value in seen:
                    tool_dup += 1
                    warnings.append(f"{cid}: tool_actions 重复 '{t.value}'")
                seen.add(t.value)

            # memory_needed 与 tool 一致性提示
            if exp.memory_needed and "retrieve_memory" not in [t.value for t in exp.tool_actions]:
                warnings.append(f"{cid}: memory_needed=true 但无 retrieve_memory tool（安全干预场景可豁免）")

    # 汇总
    result = {
        "file": str(path),
        "cases": n,
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0,
        "unique_ids": len(ids) == n,
        "tool_duplicate_cases": tool_dup,
    }

    if verbose:
        print(f"[{'OK' if result['valid'] else 'FAIL'}] {path.name}: {n} 条 case, {len(errors)} errors, {len(warnings)} warnings")
        for e in errors[:20]:
            print(f"  [ERR] {e}")
        for w in warnings[:10]:
            print(f"  [WARN] {w}")
        if len(errors) > 20:
            print(f"  ... 另有 {len(errors) - 20} 条错误")
    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python -m evaluation.validate_dataset <dataset.jsonl>")
        sys.exit(1)
    path = Path(sys.argv[1])
    result = validate_file(path)
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
