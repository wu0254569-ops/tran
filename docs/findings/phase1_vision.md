# 阶段一发现：环境与视觉能力（2026-10-12）

## 已完成
- 独立环境 `fastvlm`：torch 2.14.1+cpu、transformers 4.39.3、timm 1.0.30、peft、bitsandbytes、pillow、playwright。
  - 坑：torchvision 需从 CPU 源安装，否则 `torchvision::nms does not exist`。
- `./model`（FastVLM-0.5B）bfloat16 加载成功，可看图生成文本。
- 离线 2048 网页（`web/2048/index.html`）+ Playwright 操作脚本可截图、执行动作。
- 一轮接口 dataclass、配置加载、GameEnv 抽象已落地。

## 关键坑
- 该 checkpoint 的 `generation_config.json` 缺 `bos_token_id` / `eos_token_id`，
  自定义 `generate` 走 `inputs_embeds` 路径时会报错；需从 `model.config` 补回。

## 视觉能力实测（0.5B，greedy）

| 条件 | 结果 |
|---|---|
| 整页截图 / “读 4x4 数字” | 退化：`1,2,3,4,5,…`（数数） |
| 裁棋盘 / 同问 | `1, 2, 4, 4`（结构错） |
| 裁棋盘放大 1024 / 同问 | `1, 2, 4, 4` |
| 合成极清棋盘 / “整盘逐行” | 退化：`1 2 3 4 0 0 …` |
| 合成棋盘 / “row 1 column A” | `2` ✅ |
| 合成棋盘 / “row 1 column C” | `4` ✅ |
| 合成棋盘 / “row 2 column C” | `4` ✗（应为 2） |
| 合成棋盘 / “读第 1 行” | `2 0 4 4` ✅ |
| 合成棋盘 / “读第 3 行” | `2 0 0 0` ✗（串行） |

## 结论
- 模型**能**在清晰图上做局部读取，但**整盘读不出、且不可靠**；
  主要失效模式是**生成退化（数数）**与**串行**。
- 直接让模型“看整页图自己下棋”目前**不可行**。

## 待决策
1. 感知这一关怎么过（强约束逐格问答 / 视觉再增强 / 先监督微调会读盘 / 换基座）。
2. 核心 loss（相似度）的具体形式（`核心.md`）。
