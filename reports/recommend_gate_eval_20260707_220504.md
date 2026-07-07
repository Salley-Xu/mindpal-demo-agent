# 推荐门控评测报告

> 生成时间: 2026-07-07 22:05:04  
> 测试用例: 96 条

## 总体指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Gate Accuracy | 81.25% | 门控判断（推荐/不推荐）准确率 |
| Mode Accuracy | 67.71% | 推荐模式（hard/soft/none/safety）准确率 |
| Hard Precision | 0.00% | 硬推场景中正确命中的比例 |
| Soft Recall | 68.42% | 应 soft 推荐的覆盖度 |
| Safety Lock Rate | 100.00% | Level 3 危机场景被正确拦截的比例 |
| Level 2 Filter Rate | 100.00% | Level 2 高风险场景正确过滤比例 |
| Cooldown Accuracy | 100.00% | 冷却机制正确率 |
| None Accuracy | 100.00% | 不推荐场景正确率 |

## 分项统计

| 类别 | 总数 | 正确 | 正确率 |
|------|------|------|--------|
| Gate (推/不推) | 96 | 78 | 81.25% |
| Mode (模式) | 96 | 65 | 67.71% |
| Hard 推荐 | 13 | 0 | 0.00% |
| Soft 推荐 | 57 | 39 | 68.42% |
| Level 3 安全锁 | 6 | 6 | 100.00% |
| Level 2 过滤 | 8 | 8 | 100.00% |
| 冷却 | 3 | 3 | 100.00% |

## 错误详情 (31 条)

| ID | 场景 | 期望 | 预测 | 原因码 |
|----|------|------|------|--------|
| rg_002 | academic_stress | 推荐/soft | 不推/none | - |
| rg_004 | academic_stress | 推荐/soft | 不推/none | - |
| rg_005 | academic_stress | 推荐/soft | 不推/none | - |
| rg_007 | academic_stress | 推荐/soft | 不推/none | - |
| rg_009 | academic_stress | 推荐/soft | 不推/none | - |
| rg_010 | academic_stress | 推荐/soft | 不推/none | - |
| rg_011 | academic_stress | 推荐/soft | 不推/none | - |
| rg_013 | job_seeking | 推荐/soft | 不推/none | - |
| rg_015 | job_seeking | 推荐/soft | 不推/none | - |
| rg_016 | job_seeking | 推荐/soft | 不推/none | - |
| rg_018 | job_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,negative_trend,soft_recommendation |
| rg_019 | job_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,negative_trend |
| rg_021 | job_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,negative_trend,soft_recommendation |
| rg_031 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_032 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_033 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_034 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_035 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_036 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_037 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_038 | relationship_conflict | 推荐/soft | 不推/none | - |
| rg_061 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_062 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_063 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_064 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_065 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_066 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_067 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_068 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |
| rg_069 | explicit_help_seeking | 推荐/hard | 推荐/soft | explicit_help_seeking,high_emotion_intensity,soft_recommendation |