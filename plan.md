# Dual-Track Memory for GUI Agents

> **状态**: v3 设计(2026-05-14)。**核心 architectural shift**:implicit memory 从 text 输出改为 **latent space + LLaVA-style projector + soft prompt 注入 VLM embedding 层**;显式仍是 anchor + 结构化文本 + 递推增量更新。早期版本(v1 CFM-flow + text decoder / v2 text-only implicit)的修订理由见 §16 决策记录。

## 1. 研究问题

**如何为 GUI agent 设计能同时处理符号事实与程序性模式的 memory 机制?**

现状:long-horizon GUI agent 的失败主因是 within-task memory failures(MemGUI-Bench, arXiv 2602.06075,Liu et al. 2026 报告 PMH + ProcMH 占非超时失败 58.9%),不是 perception 或 action 错。现有 memory 机制(M3A / Chain-of-Memory / Mobile-Agent-E / AndroTMem ASM ...)**都通过文本通道注入 VLM**,信息密度受 token 离散化限制。

**核心论断**:

> Memory 应分两 medium:
> - **显式 = 结构化文本**(anchor-based,递推增量更新),负责 ground truth facts + cross-step value 持久化
> - **隐式 = 连续 latent space**,通过 LLaVA-style projector 直接注入 VLM 的 input embedding 层,绕过 text 离散瓶颈,负责行为模式 + 远期压缩 + forward 预测
> - VLM 是裁判:做 goal-vs-facts 对比与决策,memory 不越权
> - 两种 medium 在 VLM input 层汇合,token-level 并存

## 2. 核心假设

| H | 假设 | 验证手段 |
|---|---|---|
| **H1 (系统)** | Dual-medium memory(text + soft prompt 双注入)在 MemGUI-Bench 上 IRR/SR/pass@k 显著高于单 medium | 主实验 (§9.1) |
| **H2 (medium 分工)** | Explicit text 主修 PMH (fact amnesia) + OMH (via divergence);Implicit soft prompt 主修 ProcMH(loop / repeat / silent failure) | 失败模式标注 + 单 track ablation (§10.4) |
| **H3 (latent injection 价值)** | Latent → soft prompt 注入 > Latent → text decoder → 注入(信息密度优势 measurable) | path 1 vs path 2 head-to-head (§9.5) |
| **H4 (cross-channel ground truth)** | UI tree + ADB shell 互证为 explicit anchor 提供单通道方法没有的 confidence signal | divergence-on/off ablation (§9.6) |
| **H5 (开源小模型 + dual memory > 闭源大模型 + 无 memory)** | UI-TARS-1.5-7B + dual-track 的 SR 显著高于 Gemini-3-Flash baseline(若 H5 成立 → paper headline) | cross-agent 对比 (§9.4) |

**v2 → v3 变化**:H3 重新定义为 latent 注入 vs text decoder 对比(之前是 flow vs deterministic);H5 是新加的(凭借 soft prompt 路线得到的强 narrative)。

## 3. Motivating Examples

### Case 1: SetAlarmTask (单 app,中等步数)
- Goal: weekend alarm 8:25 AM, ringtone "beebeep", vibrate off
- 失败 1 (PMH): step 7 只点 Saturday 没点 Sunday → **explicit anchor `day=Saturday=False, day=Sunday=False`** 直接 surface 给 VLM
- 失败 2 (OMH): step 13 declare success 但 vibrate 还开 → **explicit divergence(UI vs ADB 不一致)+ implicit anomaly soft prompt** 双修

### Case 2: AcceptMeetingTask (单 app,短步数,silent failure)
- Goal: 回复 Daniel 邮件 "I'll be there at 10:00 AM on Thursday."
- 失败:step 4 issue input_text 时 body 字段未 focus → TYPE 静默 drop → body 空 → 发送失败,但 agent declare success
- **explicit anchor `compose.body = "Compose email"`(占位符未变)** + **implicit soft prompt 编码"TYPE 后 anchor 没变 = silent failure pattern"** → VLM 看见双重信号应能 abort

### Case 3: 跨 app 比价分享(长步数,跨屏值持久)
- Goal: 比较 JD vs 淘宝 AirPods 价格,把便宜的加购,分享给 Alice
- 失败:跨 app 切换后,VLM 忘了 JD 价格
- **explicit anchor bank Tier 2(extracted_value type)** 持久化 Price_JD / Price_Taobao,跨 app retrieve

## 4. 架构总览

```
                          t 时刻输入
              ┌─────────────────┬───────────────────┐
              ▼                 ▼                   ▼
         UI tree           action history        goal text
         ADB facts         fact deltas           m_{t-1}
         (现有 system.py)  (from explicit)       (cached)
              │                 │                   │
              ▼                 │                   │
   ┌─────────────────────┐      │                   │
   │  EXPLICIT Path      │      │                   │
   │  (text medium)      │      │                   │
   │                     │      │                   │
   │  Tier 1: scratchpad │      │                   │
   │   R1-R6 抽取        │      │                   │
   │   UI+ADB 互证       │      │                   │
   │   divergence flag   │      │                   │
   │  Tier 2: anchor bank│      │                   │
   │   5 type + links    │      │                   │
   │   P1-P5 auto promo  │      │                   │
   │   Retrieve top-K    │      │                   │
   │                     │      │                   │
   │  →递推增量更新       │      │                   │
   │   (no change=no upd) │      │                   │
   └──────────┬──────────┘      ▼                   │
              │            ┌────────────────────────┴────┐
              │            │  IMPLICIT Path              │
              │            │  (latent medium)            │
              │            │                             │
              │            │  Memory Encoder (Transformer)│
              │            │   inputs: action seq +      │
              │            │   fact_delta + goal_emb +   │
              │            │   m_{t-1}                   │
              │            │  ↓                          │
              │            │  m_t ∈ R^256                │
              │            │  ↓ (gated recurrent update) │
              │            │  m_t after update           │
              │            │  ↓                          │
              │            │  Projector (LLaVA-style)    │
              │            │  ↓                          │
              │            │  k=8 soft tokens × d_vlm    │
              │            │  (e.g., 8 × 4096)           │
              │            └──────────────┬─────────────-┘
              │                           │
              ▼                           │
       structured text                   │
       (anchor list +                    │
        divergence flags +               │
        retrieved Tier 2)                │
              │                           │
              ▼                           │
    [VLM tokenizer]                       │
              │                           │
              ▼                           ▼
   ┌─────────────────────────────────────────────────┐
   │  VLM Input Embedding Sequence:                   │
   │  [implicit_soft_tokens (k=8)]                    │
   │  + [system_text_tokens]                          │
   │  + [goal_text_tokens]                            │
   │  + [explicit_anchor_text_tokens]                 │
   │  + [history_tail_text_tokens]                    │
   │  + [image_tokens (VLM's own vision projector)]   │
   │  + [user_query_text_tokens]                      │
   └────────────────────┬────────────────────────────┘
                        ▼
        Self-host VLM (UI-TARS-1.5-7B with LoRA)
                        ▼
                   Action output
                        ▼
              Execute on emulator
                        ▼
                step t+1 (cache m_t)
```

**核心 architectural 性质**:

1. **两种 medium 互补**:explicit text(可读、可 debug、ground truth)+ implicit latent(高带宽、端到端可训、隐式压缩历史)
2. **LLaVA-style projector**:把 memory 当**第三种模态**接入 VLM(类比 vision projector)
3. **VLM 自身**:UI-TARS-1.5-7B self-host + LoRA finetune(soft prompt 注入需要 embedding 层访问)
4. **递归 latent state**:m_t = gated_update(m_{t-1}, h_t),实现远期信息压缩
5. **训练 3 阶段渐进**:Path 1 (text decoder) baseline → Alignment pretraining(text-distill)→ Action-loss + LoRA finetune

## 5. 显式 vs 隐式 — 边界规范 (v3)

| 维度 | Explicit | Implicit |
|---|---|---|
| **Medium** | 结构化文本(VLM tokenize 后吃 token embedding) | 连续向量(projector 后直接拼到 VLM embedding 序列) |
| **更新机制** | **递推增量**(no change=no update,new anchor=append,modified=in-place edit) | **每步重生成**(m_t 是 recurrent latent state,内部通过 gated update;外部 soft prompt 每步 fresh decode) |
| **看什么** | 屏幕事实 + ADB ground truth + 跨步关键状态 | 行为模式 + subarea 进度 + forward cue |
| **存什么** | (id, value, sources, confidence) 4-tuple + (type, content, evidence, links) for Tier 2 | m_t ∈ R^256(black-box,可 probe but not directly readable) |
| **训练** | 0 学习参数(R1-R6 + P1-P5 程序化触发) | encoder + projector + LoRA 端到端训(action loss) |
| **VLM 注入位置** | system / goal / explicit text section | input embedding sequence 最前,k=8 soft tokens |
| **修哪类失败** | PMH (fact amnesia)、OMH (via divergence)、cross-screen value loss | ProcMH (loop / silent fail / repeat)、long-horizon compression、forward-looking cue |

## 6. 显式轨详细设计

### 6.1 Tier 1 — Working Scratchpad(递推增量更新)

```
状态 t 的 anchor store(纯结构化文本):
  anchor_1: value=X (last seen step a, source=ui✓+adb✓ STRONG)
  anchor_2: value=Y (last seen step b, changed at step c)
  anchor_3: value=Z (newly appeared step t)
  ...

更新规则(每步):
  - 无变化 → 不动这一行
  - 已有 anchor 值变了 → 改这一行
  - 新 anchor 出现 → append 新一行
  - 之前 anchor 离开屏幕 → 标 stale 但保留(off-screen 记忆)
```

**抽取规则 R1-R6**(generic,无 per-task annotation):

| 规则 | 入选条件 |
|---|---|
| K1 可交互且状态承载 | is_clickable ∨ is_editable ∨ is_checkable 且有 state field |
| K2 ADB 持久化 | content provider / DB 字段 |
| K3 语义模式 | UI text 匹配 price/email/date/phone/time/url 正则 |
| K4 当前选中 | is_selected 或 spinner 当前项 |
| K5 Label-Value 对 | labeled_by 关系 |
| K6 AppBar Title | 当前位置 |

**互证**:per-app anchor mapping 表(每 app ~15 entry)指定 UI predicate + ADB path + decoder。Divergence 检测在 prompt 独立 section 渲染。

**复用现有 MobileWorld**:`src/mobile_world/runtime/app_helpers/system.py` 已有 ~10 个 ADB helper(用于 `is_successful`),W0 包装成统一接口供 runtime 用。

### 6.2 Tier 2 — Anchor Bank

```
Anchor = (id, type, content, evidence, links, status)
```

**5 种 type**:EXTRACTED_VALUE / IDENTIFIED_ENTITY / COMPLETED_SUBGOAL / PERSISTENT_CHANGE / EXCEPTION

**P1-P5 自动 promotion**:
- P1 EXTRACTED_VALUE: TAP 进详情 + 出现新语义模式
- P2 PERSISTENT_CHANGE: ADB 字段值变化
- P3 COMPLETED_SUBGOAL: subarea 所有 anchor 达成 + active_subarea 切走
- P4 IDENTIFIED_ENTITY: VLM thought 显式提及 + scratchpad 含该实体
- P5 EXCEPTION: 新 window 出现非主流程

**Retrieve-Reason-Update 循环**:每步 retrieve top-K=5 + all EXCEPTION;deterministic scoring(type 权重 + entity Jaccard + goal cosine + 衰减引用频率)。

**Anchor bank 容量 50**,LRU + relevance 双因素淘汰。

## 7. 隐式轨详细设计 — Latent Space + Soft Prompt 注入

### 7.1 职责(3 项)

1. **历史状态压缩**(远期 step 超出 K=15 window 的信息靠 m_t 递归保留)
2. **行为模式识别**(loop / retry / silent failure 等纯 trajectory 信号)
3. **Forward subarea 预测**(teacher-trained)

**不做**:
- 当前 fact 编码(explicit 已 cover)
- Goal-vs-facts 对比(VLM 的活)
- 具体 action 推荐(越权)

### 7.2 Architecture

```
Memory Encoder:
  inputs:
    - step_tokens (last K=15):
      step_token_i = embed(verb) ⊕ embed(target_id) ⊕ MLP(fact_delta) 
                   ⊕ embed(result)  → 256-d
    - m_prev_token: LayerNorm(m_{t-1})
    - goal_token: LayerNorm(g_emb)
  layers: 2-layer causal Transformer, d=256, heads=4
  output: h_t = last token's hidden state

Gated Recurrent Update:
  z = σ(W_z · [h_t; m_{t-1}])
  m̃ = tanh(W_m · h_t)
  m_t = (1-z) ⊙ m_{t-1} + z ⊙ m̃
  m_t = LayerNorm(m_t)         ← 防止 magnitude drift

Projector (LLaVA-style):
  v_t = W_proj(m_t) ∈ R^{k=8 × d_vlm_embed=4096}
  
Injection:
  VLM_input_embeds = [v_t[0..7]] + tokenize(text...) + vision_projector(image)
```

**参数**:
- Memory encoder: ~5M
- Projector (Linear 256 → 8×4096): ~8M
- VLM LoRA (rank 8, attention + FFN): ~50M
- **总可训 trainable: ~63M**(VLM 主体 7B frozen)

### 7.3 推理时序

```python
def implicit_step(self, action_history, fact_deltas, goal_emb):
    # encoder forward
    tokens = [LayerNorm(self.m_prev), LayerNorm(goal_emb)] + step_tokens
    h_t = self.encoder(tokens)[-1]
    
    # gated recurrent update
    z = σ(W_z @ cat([h_t, self.m_prev]))
    m_tilde = tanh(W_m @ h_t)
    m_t = (1-z) * self.m_prev + z * m_tilde
    m_t = LayerNorm(m_t)
    
    # project to soft tokens
    v_t = projector(m_t).reshape(8, 4096)
    
    # cache
    self.m_prev = m_t.detach()
    
    return v_t   # 直接拼入 VLM input embedding 序列

def reset(self):
    """新 task 开始时调,避免跨任务 contamination"""
    self.m_prev = self.learnable_m_0
```

## 8. 训练流程 — 3 阶段渐进

### Stage 0(W3 末):Path 1 baseline (text decoder) sanity check
- 跑 latent → text decoder → VLM,确认 implicit signal 有价值
- 若 SR ≈ C0 → 整个 implicit 不做

### Stage 1(W5):Alignment Pretraining(text-distill)

**目标**:让 projector 输出落在 VLM 认识的 embedding 流形

```python
for batch:
    m_t = memory_encoder(batch.context)
    v_t = projector(m_t)
    
    p_soft = VLM(prepend=v_t, text=batch.input)
    p_text = VLM(prepend=None, text=batch.input + teacher_text)
    
    loss = KL(p_soft || p_text.detach())
    # 仅更新 projector + encoder,VLM 完全 frozen
```

**关键 trick**:projector 初始化使其输出 norm 匹配 VLM token embedding 分布(防 OOD)。

**Cost**: 60k step samples × ~30s training step ≈ 8-12h on 1× A100

### Stage 2(W6):Action-loss + LoRA finetune

**目标**:让 VLM **学会使用**我们的 soft prompt

```python
# 加 LoRA (rank=8) 到 VLM attention + FFN
for batch:
    m_t = memory_encoder(batch.context)
    v_t = projector(m_t)
    
    logits = VLM_with_LoRA(prepend=v_t, text=batch.full_input)
    
    loss = NLL(logits, batch.true_action_tokens)
    loss += 0.1 * aux_losses(m_t, batch.labels)   # 多 head 辅助监督
    
    # 更新:projector + memory encoder + LoRA(~63M total trainable)
```

**Cost**:30k step samples × LoRA finetune ≈ 12-16h on 1× A100

### Stage 3(W7):Validation + 4 conditions head-to-head

| Condition | Memory medium |
|---|---|
| C0 | none |
| explicit-only | text |
| implicit-only | soft prompt |
| **dual_track** | text + soft prompt |

跑 20 task subset on UI-TARS-1.5-7B,SR / IRR / MFR / counterfactual sensitivity。

### 训练数据

| 数据 | 量 | 用途 |
|---|---|---|
| MobileWorld trajectories(自采,W3-W4)| 1500-2000 traj × ~20 step = ~30-40k samples | Stage 1+2 主 |
| Teacher-annotated text(Claude Opus,W3-W4)| ~5000 sample(oracle mode) | Stage 1 alignment 的 teacher text |
| 失败 trajectory | ~30% 占比 | 防 compounding error |

## 9. 实验设计

### 9.0 Stage 0 - Phase 1 Free-form Teacher Discovery (W3)

不限 schema,让 Claude Opus 自由产 memory text,cluster 分析浮现的"自然类别"(参 `output/test/AcceptMeetingTask/traj_annotated.json` 已示范)。结果反推 Schema 设计 + 训练 label 格式。

### 9.1 主实验:Dual-track ablation on MemGUI-Bench (W7-W8)

| Condition | explicit | implicit | 目的 |
|---|---|---|---|
| C0 | ✗ | ✗ | 下界 |
| C1 (random text control) | random ASCII | ✗ | 防 prompt-length confound |
| C2 (explicit-only) | ✓ | ✗ | text path 单独贡献 |
| C3 (implicit-only) | ✗ | ✓ | soft prompt 单独贡献 |
| **C4 (dual_track, ours)** | ✓ | ✓ | 主 claim |
| C5 (text-implicit, path 1) | ✓ | latent→text decoder | H3 验证(latent injection vs text decoder) |

跑 MemGUI-Bench 128 task on UI-TARS-1.5-7B(主 agent)。

### 9.2 Cross-agent transfer (W9)

- Qwen2.5-VL-7B(开源通用)+ dual-track  
- UI-Venus-7B / GUI-Owl-7B(开源 GUI-specialized)对比
- **Gemini-3-Flash + explicit-only**(不能用 soft prompt,但 explicit 仍可)→ 对照 closed-source 用户

### 9.3 失败模式分析

用 MemGUI-Eval(MemGUI-Bench 自带的 3-stage progressive scrutiny evaluator)对所有 trajectory 标 5 类失败(PMH / ProcMH / OMH / KD / IM)。

**期待**:C4 在 PMH(被 explicit 修)+ ProcMH(被 implicit 修)上 reduction 显著高于其它 condition;KD/IM 上无 reduction(memory 不修这两类)。

### 9.4 H5 验证:开源 + dual memory vs 闭源 + 无 memory

- UI-TARS-1.5-7B + dual-track SR
- Gemini-3-Flash baseline SR
- 若前者 > 后者 → **paper headline**:"开源 7B model with right memory beats closed-source SOTA"

### 9.5 H3 验证:latent vs text decoder

C4 vs C5 head-to-head:soft prompt 注入是否比 text decoder 注入更好?
- C4 > C5 → latent injection 突破 text bottleneck,paper main contribution
- C4 ≈ C5 → 两种 medium 等价,**仍 publishable**(诚实 finding)
- C4 < C5 → soft prompt OOD 风险未克服,paper 退回 path 1 主 claim

### 9.6 H4 验证:UI+ADB 互证价值

- C4 with full UI+ADB → SR_full
- C4 with UI-only(关掉 ADB extractor)→ SR_ui_only
- gap = ADB ground truth 的 marginal value

## 10. 评测指标

### 10.1 主指标(MemGUI-Bench 自带)

| 指标 | 含义 | 我们目标 |
|---|---|---|
| **SR** (pass@1) | 单 attempt 成功率 | 优于 baseline + 8%+ |
| **IRR** (Information Retention Rate) | 信息单元正确召回率 | explicit Tier 2 最直接体现 |
| **MTPR** (Memory-Task Proficiency Ratio) | memory-task SR / standard-task SR | 直接 measure memory specific value |
| **pass@3 SR** | 多 attempt 累计 | implicit anomaly head 在多 attempt 上更明显 |
| **FRR** (Failure Recovery Rate) | 失败后快速学习 | implicit pattern + anomaly 头测试场 |

### 10.2 诊断指标

| 指标 | 含义 |
|---|---|
| **MFR** (Memory-Following Rate) | C2/C3/C4 vs C0 上 action 不同且对齐 memory hint 的步比例 |
| **Counterfactual sensitivity** | 扰动 m_t / 改 explicit text 后 action 变化率 |
| **Linear probing on m_t** | 预测 phase / progress / loop accuracy ≥ 70% |
| **OOD sanity** | Stage 1 末 VLM 输出仍通顺英文(manual 20 sample 检查) |
| **5 类失败 reduction** | PMH/ProcMH/OMH/KD/IM 各类降幅(by MemGUI-Eval) |

## 11. 实施 / 工程

### 11.1 代码 surface

```
src/mobile_world/agents/dual_track_memory/
├── __init__.py
├── canonical/
│   ├── action.py             # CanonicalAction
│   └── observation.py
├── explicit/
│   ├── ui_extractor.py       # R1-R6
│   ├── adb_extractors/       # 包装 system.py 现有 helper
│   │   ├── registry.py
│   │   ├── deskclock.py
│   │   ├── contacts.py / sms.py / calendar.py / mail.py / ...
│   ├── anchor_mappings/      # per-app UI↔ADB 对齐表
│   ├── divergence.py
│   ├── scratchpad.py         # Tier 1,递推增量更新
│   ├── anchor_bank.py        # Tier 2 + retrieve + P1-P5
│   └── render.py             # → 结构化 text
├── implicit/
│   ├── encoder.py            # 2-layer Transformer + GRU update
│   ├── projector.py          # Linear 256 → 8×4096
│   ├── soft_prompt_module.py # 整合 encoder + projector + state cache
│   └── teacher_proxy.py      # Stage 0 path 1 decoder (用于 baseline)
├── adapters/
│   ├── base.py
│   ├── ui_tars.py            # **Primary** self-host adapter
│   ├── qwen25_vl.py          # Secondary self-host
│   ├── gemini.py             # closed-source baseline(explicit-only)
│   └── claude.py             # teacher VLM
├── training/
│   ├── stage0_path1.py       # text decoder baseline
│   ├── stage1_alignment.py   # text-distill alignment
│   ├── stage2_action.py      # LoRA + action loss
│   └── data/
│       ├── trajectory_collector.py
│       └── teacher_annotator.py
└── dual_track_agent.py       # 总编排
```

### 11.2 最小侵入修改

- `src/mobile_world/runtime/client.py`: 实现 `get_ui_forest()`(现 `:142` NotImplementedError)
- `src/mobile_world/agents/registry.py`: 注册 `dual_track_ui_tars` agent
- 既有 agent(qwen3vl / general_e2e)不动

### 11.3 LLM / VLM Provider

- **Primary agent**: **UI-TARS-1.5-7B self-host**(开源,GUI 专精,~14GB VRAM,A100-40 或 2×3090)
- **Secondary agent**: Qwen2.5-VL-7B self-host(通用,做 transfer 对比)
- **Closed-source baseline**(explicit-only):Gemini-3-Flash via API
- **Teacher**: Claude Opus 4.7 via Anthropic API
- **LLM judge**(failure mode):Gemini-2.5-Pro via Google API

### 11.4 GPU / 算力需求

| 阶段 | 配置 | 时间 | 估计成本 |
|---|---|---|---|
| Stage 1 alignment | 1× A100-40G | 8-12h | $15-20 |
| Stage 2 LoRA finetune | 1× A100-40G | 12-16h | $20-30 |
| 主实验 inference(UI-TARS self-host) | 1× A100-40G + 5 emulator | ~30h | $50 |
| API call(Gemini agent + Claude teacher + Gemini judge) | API | — | ~$300-400 |
| **总** | | | **~$400-500** |

## 12. Sanity Check Ladder(每 stage 末必过)

| Check | 阈值 | 不过怎么办 |
|---|---|---|
| W0 末:`client.get_ui_forest()` 在 5 app 上 90%+ 成功 | yes | fallback UIAutomator XML |
| W2 末:explicit Tier 1 在 SetAlarmTask 上抽出含 vibrate/day 的 anchor | yes | 检查 mapping / R 规则 |
| W3 末:互证 STRONG anchor confidence ≥ 80% | yes | per-app mapping 调整 |
| Stage 0(W3 末):path 1 SR > C0 + 5% | yes | implicit 整个不做 |
| Stage 1(W5 末):VLM 输出通顺英文 + KL(soft || text) < 0.5 | yes | OOD 修复(projector init / norm 调) |
| Stage 2(W6 末):val action accuracy > path 1 | yes | 检查 LoRA / aux loss / curriculum |
| Stage 3(W7 末):linear probing on m_t for phase ≥ 70%;counterfactual sensitivity > 25% action change | yes | memory encoder 没学到 / 没被 VLM 读 |
| W8 末:MemGUI-Bench C4 - C0 > 8% SR | yes | 退回 explicit-only main claim |

## 13. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **OOD soft prompt → VLM 输出 gibberish** | Projector init 用 VLM token embed mean/std;LayerNorm output;Stage 1 alignment pretraining;LoRA give VLM 一定 adapt capacity |
| **Frozen VLM 梯度信号弱** | Layer-wise loss(中间层 hidden 也参与);gradient scaling;Stage 1 后期就加 LoRA |
| **Action label 监督稀疏** | Aux losses(从 m_t 直接预测 anchor 完成度 / phase);verb token 重加权;curriculum |
| **Recurrent state 不稳** | LayerNorm on m_t;truncated BPTT(K=10);grad clipping;learnable m_0 |
| **Compounding error**(test 时 m_t OOD)| 训失败 traj 30%;DAgger;noise injection during training |
| **Implicit-Explicit 信息重叠** | Stage 1 teacher text 明示"不复述 anchor";Ablation 验证 super-additive |
| **Per-app ADB extractor 不全** | fallback 纯 UI MEDIUM confidence,不阻塞 |
| **UI-TARS-1.5-7B 在 MemGUI-Bench 基础 SR 低**(开源 7B 普遍 < 闭源) | Paper 用 MemGUI-Eval **5 类失败 reduction**作主指标而非 absolute SR;且做 cross-agent transfer 多个 7B 对比 |
| **W7 末 path 2(soft prompt)不 work** | 退回 path 1(text decoder) main claim;paper 报 "we tried but text dominates" 作 negative result |
| **AAAI deadline 紧张** | W4 早期就并行 Path 1 + 标注数据;W6 决策点诚实切 fallback |

## 14. 时间线(单人,2026-05-13 起)

| 阶段 | 内容 | 估时 |
|---|---|---|
| **W1-W2** | (a) `client.get_ui_forest()` 实现 (b) ADB extractor registry 包装 5 app helper (c) Anchor mapping for 5 app (d) UI-TARS-1.5-7B 部署 + LoRA 环境 (e) Gemini/Claude adapter | **2 周** |
| **W3** | Stage 0 + Phase 1 free-form teacher discovery on 5 task pilot;Schema 反推;失败模式标注 rubric;Path 1 (text decoder) 跑通 | 5 天 |
| **W4** | 主 trajectory 采集(1500-2000 traj on Qwen3-VL-8B);Teacher annotation(8000-12000 step,Claude Opus + Gemini hybrid);Path 1 baseline SR 跑 | 5 天 |
| **W5** | Stage 1 alignment pretraining(projector + memory encoder,VLM frozen) | 5 天 |
| **W6** | Stage 2 action-loss + LoRA finetune;Decision point(α/β/γ 分流) | 5 天 |
| **W7** | Stage 3 validation;dual-track ablation on 20 task subset(C0-C5) | 5 天 |
| **W8** | MemGUI-Bench docker 部署 + 128 task 全跑(UI-TARS + Qwen2.5-VL);MemGUI-Eval 失败标注 | 5 天 |
| **W9** | Cross-agent transfer(Qwen2.5-VL / UI-Venus / GUI-Owl);H5 对比 Gemini-3-Flash baseline;probing | 5 天 |
| **W10** | Paper 主体写作 + 图表 finalize | 5 天 |
| **W11** | Polish + buffer + 提交准备 | 5-7 天 |

**Critical milestone**:
- W2 末:infrastructure 通,SetAlarmTask 上 dump 出 UI+ADB anchors
- W3 末:Path 1 验证 implicit signal 有价值(Stage 0 sanity)
- W6 末:Decision point — path 2(soft prompt)是否 work
- W8 末:MemGUI-Bench 主结果 ready
- W11 末:paper 可提交

## 15. 待拍板决策(Stage 0 进入前)

**N 系列(模型 / VLM)**:
- N1. Primary agent: **UI-TARS-1.5-7B**(默认,GUI 专,AndroTMem TCR 34.55%) vs Qwen2.5-VL-7B(通用)
- N2. Teacher: **Claude Opus 4.7**(默认,贵但稳) vs Gemini-2.5-Pro(便宜 30×)
- N3. Closed-source baseline: **Gemini-3-Flash**(MemGUI-Bench SOTA among API)

**S 系列(soft prompt 设计)**:
- S1. k(soft token 数): **8** 起步,W8 扫 {4, 8, 16}
- S2. m_t 维度: **256** vs 512
- S3. LoRA rank: **8**(attention + FFN)
- S4. Projector 结构: **Linear** vs MLP(2 层)
- S5. Soft token 位置: **prepend** vs insert before image vs insert before query

**T 系列(实验设计)**:
- T1. Phase 1 free-form teacher prompt 4-5 个 variant 内容
- T2. 20 task subset 具体选哪些(S0-S3 stratified)
- T3. failure mode LLM judge prompt
- T4. SR 显著阈值(默认 ≥ 8% 相对提升)

**E 系列(显式)**:
- E1. Anchor mapping 优先 5 app:**deskclock / contacts / calendar / sms / mail**(选最实用 + 已有 system.py helper)
- E2. Tier 2 anchor bank 容量:**50**

## 16. 设计决策记录(v1 → v2 → v3)

| 决策 | v1 | v2 | v3 (本) | 理由 |
|---|---|---|---|---|
| 显式表示 | per-task constraint 状态机 | UI+ADB fact scratchpad + Tier 2 anchor bank | **同 v2** | constraint 状态机 over-engineered |
| 显式更新 | 每步重建 | (未明示) | **递推增量**(no change=no update) | user 提议,token 经济 + 稳定 |
| 隐式架构 | CFM-based flow + decoder | deterministic encoder + decoder(text 输出) | **deterministic encoder + projector + soft prompt 直接注入 VLM embedding 层** | latent injection 突破 text 离散瓶颈;信息密度优势真正落地 |
| 隐式 medium | text | text | **continuous latent → soft tokens** | LLaVA-style projector,实测 well-established |
| 主 VLM | Qwen3-VL-8B | Gemini-3-Flash | **UI-TARS-1.5-7B self-host** | soft prompt 注入需要 embedding 层访问,只能开源 self-host |
| VLM-agnostic claim | 强 claim | 强 claim | **弱化为"for self-host GUI VLMs";closed-source 仍可 explicit-only 跑** | trade-off 已诚实承认;narrative 改为 "开源小模型 + 正确 memory > 闭源大模型" |
| Anchor 概念 | 含混 | "fact"(scratchpad)+ "state anchor"(bank) | **同 v2** | 两个概念名称化 |
| Anchor promotion | 人工 | 自动 P1-P5 | **同 v2** | 填 AndroTMem 留的工程 gap |
| 训练 stages | 单阶段 + 多 head | 单阶段 | **3 stages 渐进:Path 1 baseline → Alignment pretraining → LoRA action-loss** | LLaVA-style proven recipe |
| Primary benchmark | MobileWorld | MemGUI-Bench + MobileWorld | **同 v2** | MemGUI-Bench 提供 5 类失败 + IRR/FRR + 11 baseline |
| 实验方法 | A/B/B+ours | Teacher-first methodology | **同 v2,加 H3 latent vs text 单独 ablation** | 在最关键的 architectural choice 上做 head-to-head |
| AndroTMem-Bench | 用 | 用 | **不用**(repo 未充分开源,user 验证后剔除) | external |

## 17. 当前已知技术状态(2026-05-14)

- ✅ MobileWorld 部署在 docker + KVM,5 容器并行
- ✅ SiliconFlow API + Qwen3-VL-8B 已配置
- ✅ `system.py` ~10 个 ADB content provider / sqlite helper(用于 is_successful)
- ✅ `android_env` a11y gRPC forest 抓取(未接 runtime)
- ✅ AcceptMeetingTask traj 上已示范 free-form teacher annotation(`output/test/AcceptMeetingTask/traj_annotated.json`)
- ✅ MemGUI-Bench 文档已读,Docker 一键可部署
- ❌ `client.py:get_observation(type="accessibility_tree")` NotImplementedError
- ❌ UI-TARS-1.5-7B 未部署(本地 GPU / 云租待定)
- ❌ Gemini / Claude / UI-TARS adapter 未实现
- ❌ Dual-track memory 任何组件未实现
- ❌ 1500-2000 trajectory 训练集未采
- ❌ Teacher annotation pipeline 未自动化(单条 manual 已 demo)

## 18. 决策点(stop conditions)

按 §14 时间线,以下任一触发就停下来重评:

- **W0 末**: `client.get_ui_forest()` < 50% success → 改 UIAutomator XML fallback,延期 W1 1 周
- **Stage 0 (W3 末)**: path 1 SR ≈ C0 → memory 在我们 task 上不是瓶颈,**整套 implicit 不做**,改 explicit-only paper
- **Stage 1 (W5 末)**: VLM 输出 gibberish 或 KL > 1.0 → soft prompt 训练不稳;延期 W6 1 周专门修 OOD
- **Stage 2 (W6 末)**: val action accuracy 不超过 path 1 → soft prompt 路线失败,**退回 path 1 main claim**;paper 报 "we attempted latent injection but found text bottleneck not the bottleneck"
- **W8 末**: MemGUI-Bench C4 - C0 < 5% SR → paper 换 narrative,从 "dual-medium" 改为 "cross-channel ADB augmentation"
- **W11 末**: 论文未 ready → 顺延投 EMNLP 2027 spring / NAACL

任一触发都不强推。这是 honest research 必备。
