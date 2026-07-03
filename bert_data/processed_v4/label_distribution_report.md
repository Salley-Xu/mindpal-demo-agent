# 标签分布报告

生成日期: 2026-07-02

总样本数: 1254

## cssrs_lite_level 分布

| level | 样本数 | 占比 |
|-------|--------|------|
| 0 | 352 | 28.1% |
| 1 | 317 | 25.3% |
| 2 | 419 | 33.4% |
| 3 | 166 | 13.2% |

## subject_context 分布

| 标签 | 样本数 | 占比 |
|------|--------|------|
| self_current | 1166 | 93.0% |
| third_party | 52 | 4.1% |
| negated | 19 | 1.5% |
| self_past | 14 | 1.1% |
| hypothetical | 3 | 0.2% |

## evidence_tags 分布

| 标签 | 样本数 | 占比 |
|------|--------|------|
| active_ideation | 903 | 72.0% |
| intent_signal | 603 | 48.1% |
| plan_signal | 489 | 39.0% |
| passive_death_wish | 461 | 36.8% |
| method_signal | 459 | 36.6% |
| severe_distress | 420 | 33.5% |
| urgency_signal | 208 | 16.6% |
| third_party | 62 | 4.9% |
| protective_factor | 58 | 4.6% |
| loss_of_control | 53 | 4.2% |
| negation | 45 | 3.6% |
| help_seeking | 37 | 3.0% |
| preparation_signal | 11 | 0.9% |
| discussion_context | 4 | 0.3% |

## 数据来源分布

| 来源 | 样本数 | 占比 |
|------|--------|------|
| sos-1k-fine | 1247 | 99.4% |
| sos-hl-1k | 5 | 0.4% |
| mentalglm | 2 | 0.2% |

## 标签质量分布

| 质量 | 样本数 | 占比 |
|------|--------|------|
| silver | 1247 | 99.4% |
| review | 7 | 0.6% |

## 交叉分析：level × subject_context

| level | self_current | self_past | third_party | negated | hypothetical | unclear |
|-------|-------------|-----------|-------------|---------|--------------|---------|
**0** (352) | 297 | 5 | 30 | 19 | 1 | 0
**1** (317) | 314 | 3 | 0 | 0 | 0 | 0
**2** (419) | 398 | 4 | 16 | 0 | 1 | 0
**3** (166) | 157 | 2 | 6 | 0 | 1 | 0

## 边界样本清单（review 队列）

共 7 条需人工复核：

- `sos-hl-1k-200397` level_2 | self_current | hl_low_but_high_level_tag
  > 走饭，我第一次知道你，也第一次留意到你。你知道吗？我也很想离开这个世界。这是我换抑郁症第六年了。吃药吃了五年。什么时候能够戒药，遥遥无期。。很多人无法理解我，我

- `sos-hl-1k-200991` level_0 | self_current | sos-hl-1k_supplement
  > 我害怕的想哭 害怕的颤抖 如果事情越来越严重 真的可怕 一点点安全感都没有了 对这个世界感到绝望 ​​​ 打电话给他被骂 我的心在颤抖 谁也靠不住了

- `sos-hl-1k-201200` level_2 | self_current | hl_high_need_verification
  > <a target="_blank" render="ext" extra-data="type=atname" href="//weibo.com/n/Ite

- `sos-hl-1k-201244` level_0 | self_current | sos-hl-1k_supplement
  > :)晚安

- `sos-hl-1k-201246` level_0 | self_current | sos-hl-1k_supplement
  > 再等一等好吗？�� 想象的解脱并不是真实的 是你想的

- `mentalglm-300020` level_2 | self_current | mentalglm_only_supplement
  > 饭饭，我昨晚割腕失败了。唉好难受想死都割不对地方，真的好痛啊。顺便问问有没有一起约烧炭的

- `mentalglm-300024` level_0 | self_current | mentalglm_supplement
  > 我害怕的想哭 害怕的颤抖 如果事情越来越严重 真的可怕 一点点安全感都没有了 对这个世界感到绝望 打电话给他被骂 我的心在颤抖 谁也靠不住了
