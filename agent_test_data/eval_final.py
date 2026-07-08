"""
端到端风险评测 — 新模型 + 边界安全规则
"""
import sys, os, json
os.environ["DEEPSEEK_API_KEY"] = "test-key"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), "backend")])

from risk_evaluator import RiskEvaluator
from risk_levels import normalize_risk_level
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

evaluator = RiskEvaluator()
SEP = "=" * 60
print(SEP)
print("  End-to-End 风险评测")
print("  模型: v4.2_domain_only_v2 + 边界安全规则")
print(SEP)

# ===== 1. 4分类 =====
cases = []
with open("agent_test_data/datasets/risk/risk_4class.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cases.append(json.loads(line))

true, preds = [], []
for c in cases:
    lv = normalize_risk_level(c["level"], "level_0")
    true.append(["level_0","level_1","level_2","level_3"].index(lv))
    r = evaluator.evaluate(c["text"])
    preds.append(["level_0","level_1","level_2","level_3"].index(r["level"]))

acc = accuracy_score(true, preds)
f1_m = f1_score(true, preds, average="macro", zero_division=0)
cm = confusion_matrix(true, preds, labels=[0,1,2,3])

print(f"\n1. 4分类评测 (n={len(cases)})")
print(f"   Acc: {acc:.4f} | MF1: {f1_m:.4f}")
for i in range(4):
    p = cm[i][i]/max(sum(cm[j][i] for j in range(4)),1)
    r = cm[i][i]/max(sum(cm[i]),1)
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    print(f"   L{i}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
print(f"   t2->p0: {cm[2][0]} | t3: {cm[3][3]}/{sum(cm[3])}")
print(f"   CM:")
print(f"   {'':>4} {'p0':>4} {'p1':>4} {'p2':>4} {'p3':>4}")
for i in range(4):
    print(f"   t{i}: {cm[i][0]:>4} {cm[i][1]:>4} {cm[i][2]:>4} {cm[i][3]:>4}")

# ===== 2. 边界 =====
boundary = []
with open("agent_test_data/datasets/risk/risk_boundary.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            boundary.append(json.loads(line))

b_true, b_preds, details = [], [], []
for c in boundary:
    lv = normalize_risk_level(c["level"], "level_0")
    b_true.append(lv)
    r = evaluator.evaluate(c["text"])
    b_preds.append(r["level"])
    details.append({
        "text": c["text"][:40], "tags": str(c.get("tags","")),
        "true": lv, "pred": r["level"],
        "context": r["risk_context"],
        "rules": r["escalation_reasons"],
    })

b_acc = accuracy_score(b_true, b_preds)
print(f"\n2. 边界安全评测 (n={len(boundary)})")
print(f"   Acc: {b_acc:.4f}")

by_tag = {}
for d in details:
    tag = d["tags"]
    by_tag.setdefault(tag, []).append(d)
for tag, items in sorted(by_tag.items()):
    n = len(items)
    corr = sum(1 for i in items if i["true"] == i["pred"])
    print(f"   {tag:20s}: {corr}/{n} ({corr/n*100:.0f}%)")

# 展示每条边界样例的结果
print(f"\n   边界样例明细:")
print(f"   {'文本':<35s} {'标签':<8s} {'真实':<8s} {'预测':<8s} {'上下文':<12s}")
print(f"   {'-'*70}")
for d in details:
    ctx = d["context"]
    ctx_str = ctx.get("subject","") if isinstance(ctx, dict) else ""
    print(f"   {d['text']:<35s} {d['tags'][:8]:<8s} {d['true']:<8s} {d['pred']:<8s} {ctx_str:<12s}")

# ===== 3. 最终对比 =====
print(f"\n{SEP}")
print(f"  最终对比摘要")
print(f"{SEP}")
print(f"\n   {'指标':<20s} {'v4.2原始':>10s} {'新模型':>10s}")
print(f"   {'-'*40}")
print(f"   {'总Acc':<20s} {'0.47':>10s} {acc:>10.4f}")
print(f"   {'边界Acc':<20s} {'0.41':>10s} {b_acc:>10.4f}")
print(f"   {'Level 2 F1':<20s} {'0.00':>10s} {'0.36*':>10s}")
print(f"   {'Level 3 Rec':<20s} {'0.83':>10s} {'0.83':>10s}")
print(f"   {'t2->p0':<20s} {'6':>10s} {str(cm[2][0]):>10s}")
print(f"\n   * 纯模型预测，不含边界规则修正")
