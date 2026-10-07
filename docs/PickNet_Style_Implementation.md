# PickNet-Style Sequential Frame Selection — 实现设计文档

日期：2026-09-30 ｜ 阶段：Phase 2 skeleton（未训练时编写）

> **We implement a PickNet-style sequential Pick/Drop baseline using the pre-extracted 768-dimensional frame features available in our captioning pipeline, rather than reproducing PickNet's original low-resolution glance network.**
>
> 本实现是 **PickNet-style sequential frame selection baseline adapted to our CapERA + xmodaler feature-based captioning pipeline**，**不是 official PickNet reproduction**。

---

## 1. PickNet 原论文机制（Less Is More, ECCV 2018）

- 顺序扫描候选帧（sequential frame scanning），逐帧做 **Pick / Drop 伯努利决策**；
- 输入：当前帧的 56×56 灰度缩略图与"上一被选帧缩略图"的差分图（glance-and-compare），展平后送 2 层前馈网络；
- **第一帧强制 PICK**；随后每帧依概率采样 Pick/Drop；
- 选中帧数 **video-dependent**（N_p ∈ [N_min=3, N_max]，N_max 训练中收缩），平均 6–8 帧；
- Reward：λ_l·CIDEr（语言）+ λ_v·std(选中帧特征)（视觉多样性）+ 越界罚 R⁻=−1；
- 训练三阶段：①监督（XE + schedule sampling 训 captioner）→ ②强化（冻结 captioner，REINFORCE + self-critical 基线训 PickNet）→ ③适应（近似联合训练）；
- 选中帧送入 Encoder-Decoder captioner 生成描述。

## 2. 我们复现的机制（What we reproduce）

1. **顺序扫描**：t=0,1,...,9 依次处理候选帧；
2. **逐帧 Pick/Drop 决策**：决策网络输出 2 logits → P(PICK)/P(DROP)；
3. **第一帧强制 PICK**：selection_mask[:,0] ≡ 1，禁止 DROP；
4. **video-dependent 选中数量**：不同视频得到不同 N_selected（不是固定 8，也不是 Top-K）；
5. **选中帧 → captioner**：selected features（padding + attention mask）送入 xmodaler Transformer captioner；
6. **selection trace**：记录每个视频的 selected_indices / selection_probs / num_selected，供后续预算分析。

## 3. 不复现的机制（What we do NOT reproduce）

| 原论文机制 | 本 skeleton 的处理 |
|---|---|
| 灰度 glance 差分输入（56×56） | **直接用预提取的 MaxViT-S 768-d 帧特征**（与 captioner 特征统一，不重建 image-level 管线） |
| REINFORCE + self-critical baseline | **v1 不用 RL**：XE 端到端 + Straight-Through Estimator（见 §9） |
| CIDEr 语言 reward / 视觉多样性 reward | v1 不实现 reward；captioning 信号经 XE loss 梯度回传 |
| N_min / N_max 越界罚 | 以可微"期望选中率"正则项替代（见 §9） |
| 三阶段训练（监督→强化→适应） | v1 单阶段联合 XE 训练 |
| LSTM 编码器 / GRU 解码器 | 保留 xmodaler Transformer Encoder/Decoder（与 uniform baseline 完全同骨架） |

## 4. 与原始 PickNet 的差异总结

- 输入表示：768-d 视觉特征 vs 灰度差图（**刻意为之**——本实验已有特征管线，工程简单且与 captioner 输入同源）；
- 训练信号：XE + STE vs REINFORCE + CIDEr reward（**刻意为之**——v1 目标是最小可训练 skeleton，RL 协议留待后续在统一接口上替换）；
- 帧数约束：软正则 vs 硬区间罚；
- captioner：Transformer vs LSTM/GRU（与项目 uniform baseline 一致，保证可比性）。

## 5. 当前候选帧协议

- 每视频 **10 个候选帧**（2 fps 均匀采样自 5s 视频）→ `features/CapERA/maxvit_small_tf_224/{split}/{int_id}.npz` 内 `features: (10, 768)`；
- Dataset（capera.py）**只提供候选特征**：picknet 配置中 `MAX_FEAT_NUM=10`，`_sample_frame` 不触发截断，ATT_FEATS 以 (B,10,768) 进入模型；
- **Selector 是模型内部模块**，Dataset 不含任何选择逻辑；uniform-8 路径（baseline 配置）保持原样不动。

## 6. Selector 架构（v1）

```
x_t (768) ──┐
            ├─> GRUCell(768 → 256):  h_t = GRU(x_t, h_{t-1})     (sequential state)
h_{t-1} ────┘
[h_t ; x_t] (1024) → Linear(1024→256) → ReLU → Linear(256→2) → logits_t
p_t = softmax(logits_t)[1]   # P(PICK)
t = 0: 强制 p_0 = 1
候选掩码外的位置（padding）：p_t = 0
```

- 决策依赖 previous observations（GRU 状态）与当前帧——**不是逐帧独立 Top-K 打分**；
- 无 attention、无 global memory、无 utility head（后续方法在此接口上扩展）。

## 7. 顺序决策机制

- 训练：`hard_t ~ Bernoulli(p_t)`，**STE**：前向用 hard（0/1），反向梯度经 `p + (hard − p).detach()` 流回 p；
- 推理：确定性 argmax `hard_t = (p_t ≥ 0.5)`；
- `hard_0 ≡ 1`（第一帧）；
- `S = {t : hard_t = 1}`；选中特征按原时间序保留（不重排，与 PickNet 扫描序一致）。

## 8. Variable-length 处理

- 选中帧数量 N_selected ∈ [1, 10]，batch 内各视频不同；
- 统一表示：`selected_features = feats * hard.unsqueeze(-1)`（drop 位置置零），shape 恒为 (B,10,768)；
- `ATT_MASKS ← selection_mask (B,10)`（选中=1）→ xmodaler 的 `get_extended_attention_mask` 自动生成 `EXT_ATT_MASKS`，TransformerEncoder 对未选位置完全不注意（该 mask 机制已在 Phase 1 审计确认，`transformer_enc_dec.py:53-90`）；
- 等价于"padding + attention mask"形式：Video A 选 3 帧 = [x_a, x_b, x_c, 0, ..., 0] + mask [1,1,1,0,...,0]。

## 9. 训练策略（v1）

- **单阶段端到端 XE**（LabelSmoothing 0.1，同 baseline 配置），captioning 梯度经 STE 进入 selector；
- **期望预算正则**（可微，非 RL reward）：
  `L_budget = λ_b · ( mean_video(mean_t p_t) − target_ratio )²`，默认 λ_b=1.0、target_ratio=0.5（≈ 平均 5 帧）。
  **为什么必须有它**：没有预算压力时，"全选"是 XE 损失下的平凡最优（更多帧=更多信息），selector 会退化为全 1 掩码；该正则是最小机制，使 Pick/Drop 决策有实际意义。**这是工程性 budget regularizer，不是 PickNet 的 RL reward，也不替代未来的 RL 协议。**
- Stage A（监督/启发式预热）：**跳过**——无可靠 frame-level 标签，不伪造 ground-truth pick 标签（遵循用户要求）；
- 明确声明：**本训练协议 ≠ 原论文三阶段 RL protocol**。

## 10. 评测协议

- 与 uniform baseline 完全一致：beam 5、COCOEvaler（BLEU-4/METEOR/ROUGE-L/CIDEr）、同一 val/test split 与参考 JSON；
- 额外输出 selection trace：`{output_dir}/results/{epoch}_selection_trace.json`，每视频：
  `{video_id, candidate_indices, selected_indices, selection_probs, num_selected}`。

## 11. Selection statistics（训练与评测均记录）

1. average / min / max / **median** num_selected；
2. selection ratio（avg N_selected / 10）；
3. selector inference overhead（μs/视频，含 GRU 10 步 + MLP，与 captioner 生成耗时并列记录）；
4. total inference latency（特征已预提取，口径 = selector + captioner beam 生成）；
5. 训练期每步的 avg_selected / budget_loss 写入 EventStorage（tensorboard + metrics.json）。

## 12. Known limitations（v1 明确声明）

1. 非原论文 RL 协议（无 CIDEr/diversity reward，无 self-critical baseline）；
2. STE 近似（hard 决策的可微化有偏）；
3. 预算控制是软正则，非硬约束（可能出现少量 1 帧或 10 帧的视频）；
4. 无 stop/lookahead 机制（扫描完 10 个候选才结束）；
5. 候选池仅 10 帧（2 fps）——更细候选需重提取特征（脚本已支持 --fps）；
6. 第一帧强制 PICK 是启发式（原论文也是，但我们的"第一帧"= 视频第 0 帧采样点）。

---

## 接口契约（后续 Adaptive Selector 必须遵守）

```
models/selectors/base_selector.py

class BaseSelector(nn.Module):
    def forward(self, feats, masks):
        '''
        feats: (B, T, D)  候选帧特征（D=768）
        masks: (B, T)     候选有效掩码（1=有效候选）
        returns dict:
            selected_features: (B, T, D)  drop 位置置零
            selection_mask:    (B, T)    1=选中
            selection_probs:   (B, T)    P(PICK)（含强制首帧=1）
            selection_indices: list[list[int]]  每样本选中帧下标
            num_selected:      (B,)
            num_candidates:    (B,)
            stats: dict (avg/min/max/median num_selected, selection_ratio)
        '''
```

- captioner 对 selector 无感知：它只看到 selected_features + selection_mask；
- 未来 `models/selectors/adaptive_selector.py` 只需实现同一接口并在 config 里换 `SELECTOR.NAME`。

---

## 13. 执行结果（2026-09-30：skeleton + 小规模训练）

### 冒烟测试（scripts/smoke_test_picknet.py，7/7 通过）

1. Dataset 输出完整 10 候选帧（(10,768)，无 uniform-8 截断）；selector 输出 shapes/mask/零填充全部正确；
2. 顺序决策证明：首帧必选、计数 ∈[1,10]、变长计数 [2..8] 在随机采样下出现（非固定 8、非 Top-K）；
3. XE forward 全链路 (4,26,1891) 无 NaN；
4. backward：selector GRU 收到非零有限梯度，optimizer.step 生效；
5. 4 视频 overfit：loss 4.09→2.80（平均选帧 4.9，预算正则生效）；
6. beam 推理正常；
7. 完整 COCO 评测链路（74 视频 val）跑通并输出指标与 selection summary。

### 小规模训练（10 epochs，全 train 集，其余超参同 baseline）

- **训练期选择行为**（随机采样）：avg_frames 4.98、ratio 0.498（预算正则精确收敛到目标 0.5）、median 5、范围 [2,8]；
- **推理期选择行为**（argmax）：test 上 avg **2.55 帧**、median 2、ratio 0.255——selector 学会了"少选帧"；
- **指标**（beam 5）：

| 模型 | epochs | 平均帧数 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|
| Uniform-8 baseline | 50 | 8.0（固定） | 0.170 | 0.185 | 0.388 | 0.610 |
| **PickNet-style（10ep）** | **10** | **2.55** | **0.178** | **0.184** | **0.396** | **0.652** |

- 结论：**10-epoch PickNet-style 用平均 2.55 帧达到/超过 50-epoch uniform-8 的 test 指标**（CIDEr 0.652 vs 0.610），初步验证 "less is more" 信号。⚠️ 公平对比需 50-epoch PickNet-style 全量训练（待用户确认后执行）。
- **Selector 推理开销**（val 74 视频实测）：selector-only **130 μs/视频**；preprocess 2.9 ms；beam 生成 7.0 ms；总推理 9.9 ms/视频；selector 占比 **1.3%**。
- 产物：`experiments/capera_picknet_style/`（model_final.pth、`results/{5,10}.json`、`results/{5,10}_selection_trace.json`——trace 含 video_id/selected_indices/selection_probs/num_selected）。
