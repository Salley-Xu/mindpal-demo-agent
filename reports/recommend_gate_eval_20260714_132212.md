# 推荐门控评测报告

> 生成时间: 2026-07-14 13:22:12  
> 测试用例: 96 条

## 总体指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Gate Accuracy | 95.83% | 门控判断（推荐/不推荐）准确率 |
| Mode Accuracy | 95.83% | 推荐模式（hard/soft/none/safety）准确率 |
| Hard Precision | 100.00% | 硬推场景中正确命中的比例 |
| Soft Recall | 92.98% | 应 soft 推荐的覆盖度 |
| Safety Lock Rate | 100.00% | Level 3 危机场景被正确拦截的比例 |
| Level 2 Filter Rate | 100.00% | Level 2 高风险场景正确过滤比例 |
| Cooldown Accuracy | 100.00% | 冷却机制正确率 |
| None Accuracy | 100.00% | 不推荐场景正确率 |

## 分项统计

| 类别 | 总数 | 正确 | 正确率 |
|------|------|------|--------|
| Gate (推/不推) | 96 | 92 | 95.83% |
| Mode (模式) | 96 | 92 | 95.83% |
| Hard 推荐 | 13 | 13 | 100.00% |
| Soft 推荐 | 57 | 53 | 92.98% |
| Level 3 安全锁 | 6 | 6 | 100.00% |
| Level 2 过滤 | 8 | 8 | 100.00% |
| 冷却 | 3 | 3 | 100.00% |

## 错误详情 (4 条)

| ID | 场景 | 期望 | 预测 | 原因码 |
|----|------|------|------|--------|
| rg_004 | academic_stress | 推荐/soft | 不推/none | - |
| rg_007 | academic_stress | 推荐/soft | 不推/none | - |
| rg_009 | academic_stress | 推荐/soft | 不推/none | - |
| rg_016 | job_seeking | 推荐/soft | 不推/none | - |