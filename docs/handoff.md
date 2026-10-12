# 交接文档（自然语言模型自定义 loss 循环实践）

> 目的：把现状、问题、目标、可协作接口与计划写清楚，便于**更换更好的模型**后无缝接续。
> 仓库：`/home/play/build/forward`　环境：conda `fastvlm`　最近提交：`87e6a72`

---

## 一、现状（已跑通的部分）

### 1. 规格（提示词）
- `系统提示词/`（**工作版 v0.5.0**）：`系统提示词.json`、`模型输出json.json`、`描述.md`。
- `原始系统提示词/`（**原稿，勿改**）：`描述.txt`、`核心.md`、两个 json 原稿。
- 全部为**合法 JSON**（原稿曾用中文弯引号导致解析失败）。

### 2. 环境
- conda 环境 **`fastvlm`**（由 `scripts/setup_env.sh` 创建）。
- 关键版本：`torch 2.14.1+cpu`、`transformers 4.39.3`、`timm 1.0.30`、`peft 0.11.1`、`playwright 1.63`、`numpy 1.26.4`。
- 机器：Linux，**无 GPU**，8 核，内存约 7GB。

### 3. 模型
- `./model` = **FastVLM-0.5B**（`LlavaQwen2ForCausalLM`，Qwen2-0.5B + MobileCLIP-L 视觉塔）。
- 当前以**纯文本**方式使用（视觉模块保留、默认关闭）。
- 已知坑：该 checkpoint 的 `generation_config.json` 缺 `bos_token_id`/`eos_token_id`，
  自定义 `generate` 走 `inputs_embeds` 路径会报错；代码里已从 `model.config` 补回。
- 已知坑：`generate` 从 `inputs_embeds` 生成时，**返回序列不含 prompt**，直接 `decode(out[0])`。

### 4. 游戏与操作脚本
- `web/2048/index.html`：自建**离线可复现** 2048（`?seed=`），键盘←↑→↓驱动，暴露 `window.game2048`。
- `src/game/game2048_env.py`：Playwright 无头驱动，产出**文本状态**（可切图片）。

### 5. 代码骨架（英文路径，分层）
```
configs/default.yaml        所有超参（不硬编码）
src/config.py               配置加载（点号路径）
src/loop/schema.py          一轮接口 dataclass（稳定契约）
src/loop/round.py           单轮闭环：决策→执行→反思→反传→记忆
src/game/base.py            GameEnv 抽象
src/game/game2048_env.py    2048 环境（Playwright）
src/model/vlm.py            模型封装：加载/LoRA/生成/自定义 loss 立即更新
src/loss/similarity_loss.py 核心相似度 loss
src/memory/store.py         短期(缓存)/长期记忆
src/io/state_text.py        状态→文本
src/io/parse.py             宽松 JSON / 动作解析
scripts/                    setup_env / run_round / test_train_step / diag_* / smoke_model
docs/findings/              阶段发现
```

### 6. 已验证
- 模型可加载、可纯文本生成。
- 单轮闭环机械上跑通：`决策→环境→反思→(微调)→记忆`。
- **核心 loss 有效**：`L = 1 - cos_sim(h_wrong, emb(right))`，5 步内 cosine 上升
  （`down: +0.048`，`left: +0.042`），基础权重与**语义特征层(embedding/lm_head)冻结**，只训 LoRA（108 万参数，0.17%）。

---

## 二、现有问题（按优先级）

1. **模型不吐结构化 JSON**（最关键）。0.5B 只输出散文（`I need to move up ...`），
   导致动作走兜底、反思字段全空、**微调几乎不触发**。→ 机器通了，模型输出不可控。
2. **视觉读盘不可靠**。整盘读取会退化成数数（`1 2 3 4 …`），逐格提问约五成正确。
   已因此**切纯文本**规避。
3. **模型棋力弱**。默认动作常为 `up`，反复无效移动、棋盘停滞，无学习信号。
4. **输出 key 语言**：模板暂用英文 key（`self_action`/`right_action`），与 spec 中文 key 不一致。
5. **硬件受限**：无 GPU、内存 7GB，反传需靠 LoRA；全参训练仅预留接口。
6. **未实现项**：反向随机性（约定 3 轮内）、长期记忆 schema、约束解码/重试。

---

## 三、目标

- **近期**：让模型输出**可控可解析**（动作 + 反思字段稳定），闭环能持续产出有效微调样本。
- **中期**：跑多局，统计**得分随训练轮次的变化曲线**（唯一评价指标，不做 OpenAI 对照）。
- **长期**：换 **Qwen 系列大模型**、游戏换**宝可梦**（接收端兼容层，之后向接收端 API 传图片+文本）。

---

## 四、可协作（接口与替换点）

> 设计原则：**新框架只替换实现，不改接口**。

| 想换什么 | 改哪里 | 说明 |
|---|---|---|
| **模型** | `src/model/vlm.py` + `configs/default.yaml` | 保持 `encode_messages / generate_ids / action_token_id / train_step / action_similarity` 语义 |
| 游戏 | 实现 `src/game/base.py` 的 `GameEnv` | `reset/step/observe/actions` |
| 感知模式 | `configs/default.yaml: perception.mode` | `text` ｜ `image`（视觉代码已在，未启用） |
| loss | `src/loss/similarity_loss.py` + `loss.*` 配置 | `metric`(cosine/dot/mse)、`direction`(pull/push) |
| 记忆 | `src/memory/store.py` | 短期(缓存，全量覆盖)/长期(读写/模糊匹配) |
| 循环 | `src/loop/round.py` | 一轮接口见 `src/loop/schema.py` |

**换模型的注意事项**
1. 动作词必须是**单 token**（Qwen 系：`" up"=705 /" down"=1495 /" left"=2115 /" right"=1290`）；换模型需重算。
2. 需保留 chat template（`apply_chat_template`）。
3. 若模型无远程代码，可去掉 `trust_remote_code`，并复用 `_patch_generation_config` 逻辑。
4. 语义特征层冻结 + 仅 LoRA 训练，是内存与“动作不塌缩”的双重保障，建议保留。

---

## 五、计划书（建议顺序）

**P0　结构化输出可靠性**（先做，否则后续无意义）
1. 动作：改为**四选一**（只允许输出 `up/down/left/right` 之一）。
2. 反思：**分步提问或给死模板填空**，逐个拿字段（结果评价 / 错误操作 / 应有操作 / 是否微调）。
3. 加**解析失败重试**；必要时上**约束解码**。
4. 统一 key 语言（中文 spec 或英文实现，二选一并回写 spec）。

**P1　跑通稳定闭环并出曲线**
5. 多局多次，记录每局得分、微调次数、loss/cosine 变化，产出**得分曲线**。
6. 观察“模型自决是否微调”的比例是否合理（避免一直不微调）。

**P2　补齐约定功能**
7. **反向随机性**（前向 dropout + 反向少量随机性，强度≈微调；无现成实现可跳过）。
8. **长期记忆 schema**（由我们定，强约束）：`{op: write|update|delete, key, value}` + key 索引 + 模糊检索。
9. 可选：恢复**图片/图文混合**感知（换更强模型后重测读盘）。

**P3　扩展**
10. 换 Qwen 大模型 / GPU 全参训练接口；游戏换宝可梦（接收端兼容层）。

---

## 六、关键决策记录（勿重复追问）

- 每轮**两次模型调用**：先决策，看到结果后反思 + 记忆。
- 感知：**当前纯文本**，视觉模块**保留但关闭**。
- 操作：**单字、单 token**；动作空间不写死（2048 为 up/down/left/right）。
- loss：`L = 1 - cos_sim(h_wrong, emb(right))`，**拉近**；**不**叠加模型自报 loss（反思已含 loss 成分）；
  **冻结 embedding/lm_head（语义特征层）**，避免不同操作被学成相似；反传范围**由我们定**（现在只训 LoRA）。
- 更新时机：模型决定微调后**立即反传**。
- **是否微调由模型自决**：简单经验走长期记忆；可抽象/可发展者才微调。
- 短期记忆 = **模型缓存**，模型自选内容、**全量覆盖**写入 json，进入下一轮；首轮含完整规则。
- 长期记忆：读取由**模型主动输出指令**触发，系统每轮附带 key 索引，**允许模糊匹配**；写入 schema **由我们强约束**。
- 评测：**只看得分**，不做 OpenAI API 对照。
- 工程：代码英文路径/文件名，中文注释尽量避免，单行 ≤ 100 字符，配置优先。

**尚未定（下一模型应向用户提填空）**
- 结构化输出强约束的具体做法（提示词 / 约束解码 / 四选一）。
- 输出 key 用中文还是英文。
- 反向随机性的具体形式；长期记忆 key 结构。

---

## 七、复现与运行

```bash
source /home/play/miniconda3/etc/profile.d/conda.sh && conda activate fastvlm
cd /home/play/build/forward

# 冒烟：模型看图/生成
python -m scripts.smoke_model --image web/2048/smoke.png

# 验证核心 loss（应看到 loss 下降、cosine 上升）
python -m scripts.test_train_step

# 跑单轮闭环（纯文本）
python -m scripts.run_round --rounds 3

# 截一张 2048 图
python -m scripts.capture_2048 --out web/2048/smoke.png --moves left,down
```

**产物**：短期记忆 `logs/memory/short_term.json`、长期记忆 `logs/memory/long_term.json`、LoRA `logs/lora/`。

---

## 八、风险与注意事项

- **内存**：反传时注意只留必要梯度（已冻结基础权重/语义层、单步损失）；更长的 prompt 会增大显存/内存压力。
- **重复动作停滞**：2048 中同方向无效移动会导致棋盘不变，注意让策略/兜底有变化。
- **模型退化**：小模型易进入数数/重复模式；强约束与重试是必需的。
- **依赖版本敏感**：`transformers` 必须 4.39.x（模型远程代码）；`peft` 用 0.11.1（新版需更高 transformers）；
  `torchvision` 必须从 **CPU 源**装，否则 `torchvision::nms does not exist`。
- **不要给模型联网模块**（避免检索最优解）。
