# PEEK 官方代码审计报告

日期：2026-09-30 ｜ 状态：只读审计（clone + 读码 + 依赖检查），**未修改 PEEK 源码、未修改现有项目、未安装任何新依赖**

---

## 1. Repository

| 项 | 值 |
|---|---|
| URL | https://github.com/momentslab/peek |
| 本地位置 | `third_party/peek`（git clone，`.git` 保留） |
| Commit | `a0df9fe0ddff4682adde9d08d77504dccd1e4ede`（2026-08-19，"update bibtex citation"，branch `main`，与 origin/main 同步） |
| License | 代码 Apache-2.0；权重 CC-BY-NC-SA-4.0（**非商用**，存于 HF Hub `momentslab/peek`，不在仓库内） |
| Paper | "PEEK: Picking Essential frames via Efficient Knowledge distillation"，arXiv:2605.31029 |
| Conference | BMVC 2026 |

## 2. Directory structure

```
third_party/peek/
├── README.md, LICENSE, pyproject.toml
├── configs/peek_base.yaml          # released-model 训练配置
├── scripts/
│   ├── prepare_manifest.py         # ActivityNet 标注 → JSONL manifest
│   ├── extract_frames.py           # ffmpeg 按 fps 抽帧 JPEG（DEFAULT_FPS=2.0）
│   ├── compute_teacher_targets.py  # Stage 1：SigLIP2 教师打分
│   ├── precompute_embeddings.py    # Stage 2 输入：MobileCLIP2 帧嵌入缓存
│   ├── train.py                    # Stage 2 训练入口（selector only）
│   └── infer.py                    # 单视频推理 CLI
└── src/peek/
    ├── data.py        # SegmentRecord（video_id/segment_id/起止秒/caption/video_path）+ manifest I/O
    ├── frames.py      # ffmpeg 抽帧
    ├── teacher.py     # SigLIP2 SO400M 教师（caption-conditioned 帧打分，仅训练期监督）
    ├── encoder.py     # 冻结 MobileCLIP2 视觉塔 + 嵌入预计算
    ├── dataset.py     # PeekSegmentDataset（读预存嵌入 + 教师 target）
    ├── model.py       # PeekScorer（selector 本体）
    ├── losses.py      # ListMLE listwise 排序损失
    ├── selection.py   # stratified_argmax / topk / uniform
    ├── inference.py   # video → 抽帧 → 编码 → 打分 → 选 k 帧（端到端）
    └── train.py       # selector 训练循环
```

## 3. Execution pipeline（真实代码链路）

```
[训练期]
ActivityNet 标注 JSON ──prepare_manifest.py──> manifest.jsonl（segment 级，含 caption）
video mp4 ──extract_frames.py (ffmpeg, 2 fps)──> frames/{video_id}/{segment_id}/frame_*.jpg
   │
   ├─ teacher.py: SigLIP2 SO400M（image×text 双塔）
   │     caption → text emb；每帧 → image emb；L2 归一化余弦 → 每 segment min-max 归一
   │     → teacher/{split}_targets/{video_id}/{segment_id}.json（frame_targets）
   │
   └─ encoder.py: MobileCLIP2-S0（冻结视觉塔，512-d）
         → embeddings/{video_id}/{segment_id}.pt {frame_indices, embeddings (N,512) fp16}
train.py: PeekSegmentDataset(embeddings+targets, 增广) → PeekScorer → ListMLE → AdamW/cosine
[推理期]
video ──frames──> MobileCLIP2-S0 逐帧编码 ──> PeekScorer → (B,T) scores
        ──selection.stratified_argmax(scores, k)──> selected_indices（k 帧，时序 bucket 内取最高分）
        ★ 仓库在此结束 —— caption 生成与下游评测【不在仓库内】
```

## 4. Selector mechanism（PeekScorer）

- **文件**：`src/peek/model.py:41 PeekScorer`
- **forward(embeddings (B,T,D), mask (B,T) bool)**：LayerNorm → Linear(D→256) → **Sinusoidal 位置编码** → **2 层 TransformerEncoder**（temporal self-attention，`src_key_padding_mask=~mask`）→ Linear(256→1) → **(B,T) scores**（released 设置为 `output_activation="identity"`，排序即得排名）
- **训练目标**：`losses.py listmle_loss`（listwise 排序损失，只依赖教师分数的相对顺序）；教师 = SigLIP2 caption-帧余弦相似度（**caption 只在训练期当监督，推理期无 caption/text**）
- **选择规则**：`selection.py stratified_argmax(scores, k)`——把时序均匀分成 k 个 bucket，每 bucket 取最高分帧 → **k 是调用方给定的固定预算（per-call fixed）**；`--k` 默认 1；备选 topk / uniform
- **Temporal modeling**：有（位置编码 + self-attention）；**adaptive budget**：无（k 固定，无停止/效用机制）
- **参数量**：~1.7M（config 注释）
- **权重**：`peek_base.safetensors`（HF，CC-BY-NC-SA），`load_peek_from_checkpoint(embedding_dim=512, ...)` 容忍键匹配加载

## 5. Captioning pipeline

**仓库内不存在 captioning 模型。** 证据：

- `grep caption src/peek/*.py`：caption 仅出现在 manifest 记录与 teacher 文本编码——**无任何生成代码**；
- README Release plan 明示："The full downstream captioning evaluation pipeline used for the paper tables is **still being prepared**"，未发布项含 ActivityNet Captions / MSR-VTT 测试评测代码；
- `scripts/infer.py` 的输出是 selected_indices/scores/timings，**没有 caption**；
- 论文实验中的 captioner 是外部 VLM（下游模型），不在本仓库。

## 6. Interface with my project

| Component | PEEK | My Project | Compatibility |
|---|---|---|---|
| Dataset | ActivityNet 段级 JSONL manifest | CapERA（video 级 JSON，10 候选帧/视频） | 需 manifest adapter（或绕过其数据管线） |
| Frame input | JPEG 帧（ffmpeg 2 fps）→ MobileCLIP2 编码 | **预提取 MaxViT-S npz (10,768)** | ✅ 我们已有特征级协议，无需走其抽帧 |
| Feature dimension | 512（MobileCLIP2-S0/S2） | 768（MaxViT-S） | ✅ `PeekScorer(embedding_dim=...)` 为构造参数，768 直接可用 |
| Selector input | (B,T,D) embeddings + bool mask | [B,10,768] + mask | ✅ **直接兼容**（embedding_dim=768） |
| Selector output | (B,T) scores → stratified_argmax(k) → indices | xmodaler 需要 selected feats + mask | ⚠️ 需薄 adapter：indices → mask/zero-pad（等价于我们 `BaseSelector` 协议） |
| Captioner | **无（外部 VLM）** | xmodaler Transformer（已跑通） | PEEK 不提供 captioner → 用我们的 xmodaler |
| Training | selector-only：ListMLE + AdamW 2e-4 + cosine + bf16 | xmodaler XE（+ 我们的 selector 训练协议） | 各自独立；可并用 |
| Evaluation | spearman / topk_recall@k / ndcg@k（排序质量） | BLEU-4 / METEOR / ROUGE-L / CIDEr（caption 质量） | PEEK 缺 caption 评测 → 用我们的 COCOEvaler |

**回答"PEEK selector 输出能否直接转成 xmodaler 输入"**：能。scores → `stratified_argmax(scores, k)` → indices → 构造 selection_mask（(B,10) 0/1，选中=1）→ `selected_features = feats * mask.unsqueeze(-1)` → 即我们 xmodaler 侧的标准 `ATT_FEATS + ATT_MASKS` 输入（与 `models/selectors/base_selector.py` 契约一致，Phase 2 已在该协议上跑通）。

## 7. 依赖检查（只记录，未安装）

| 依赖 | 要求 | 当前环境 | 状态 |
|---|---|---|---|
| python | ≥3.10 | 3.10.13 | ✅ |
| torch | ≥2.1 | 2.1.2+cu121 | ✅ |
| numpy / scipy / pillow / pyyaml / tqdm | – | 1.26.4 / 1.15.3 / 12.3 / 6.0.3 / 4.70 | ✅ |
| huggingface_hub / safetensors | ≥0.23 / ≥0.4 | 0.36.2 / 0.8.0 | ✅ |
| **open_clip_torch** | ≥2.30 | **MISS** | ❌ 缺失（MobileCLIP2 编码器需要） |
| **imageio_ffmpeg** | – | **MISS** | ❌ 缺失（抽帧需要；系统 PATH 亦无 ffmpeg） |
| **transformers** | **≥4.49** | **4.36.2** | ⚠️ **版本不足**（SigLIP2 教师 `google/siglip2-so400m-patch14-384` 在 4.36 不可用；`Siglip2Model`/FixRes 处理需要新版） |

按指示：以上缺口**仅记录，未安装、未升级**。

## 8. 结论：PEEK 能否作为论文的"母代码框架"

- **Q1 完整 pipeline？** 否——PEEK 仓库只到"selected frames"为止，**caption 生成与评测明确未发布**；
- **Q2 selector↔captioner 连接？** 代码中不存在该连接（输出 indices，captioner 是论文实验中的外部 VLM）；
- **Q3 能否接受预提取特征 [B,T,D]？** 能——`PeekScorer` 直接消费 embeddings（训练走 `PeekSegmentDataset` 的 .pt 缓存，推理也支持预计算路径），[B,10,768] 只需 `embedding_dim=768`；
- **Q4 各自 backbone？** PEEK = MobileCLIP2-S0（学生，512-d，冻结）+ SigLIP2 SO400M（教师，仅训练期）；我们 = MaxViT-S（768-d）。不改任何一方。

因此正确的角色定位：**PEEK 是"selector 参考实现/组件来源"，不是可直接承载完整 captioning 实验的母框架**。我们的母框架仍是 xmodaler（已跑通训练/评测）。对论文题目 "Adaptive Frame Pruning with Temporal Context for Video Captioning" 而言：PEEK 的"temporal context"（位置编码+2层 transformer）与蒸馏式帧打分正是可对标的模块；但 **PEEK 的 k 是固定预算、无 adaptive budget**——这是题目中 "Adaptive" 部分必须由我们自己的方法提供（PEEK 官方代码对此无现成支持）。

## 9. Required modifications（下一阶段可能需要的修改点，本阶段不做）

```text
我们的候选特征 [B,10,768] (MaxViT-S npz)
        ↓
PeekScorer(embedding_dim=768) 或 自研 selector
        ↓  scores → selection（薄 adapter：indices → mask/zero-pad）
        ↓
[未来研究：Temporal Context 增强 / Contribution Learning / Adaptive Budget]
        ↓
xmodaler ATT_FEATS + ATT_MASKS（BaseSelector 协议，Phase 2 已跑通）
        ↓
xmodaler captioner（不变）
```

具体待改文件清单（仅列出，暂不动）：
1. 新增 `models/selectors/peek_scorer_selector.py`（把 `PeekScorer` 包成 `BaseSelector` 契约，embedding_dim=768）或直接复用 `third_party/peek/src/peek/model.py` 的 `PeekScorer` 类（import 即可，其无重依赖）；
2. 若复现 PEEK 训练（ListMLE + 教师）：需先解决 `transformers≥4.49`（当前 4.36.2 不满足、按约束不升级）与 `open_clip` 缺失——或改用我们已有的 caption 数据设计蒸馏目标；
3. 若走 PEEK 数据管线：需写 CapERA → JSONL manifest adapter（`prepare_manifest.py` 目前只解析 ActivityNet 格式）；
4. `configs/capera/` 增加以 PEEK-scorer 为 selector 的实验配置；
5. 评测沿用我们的 COCOEvaler（PEEK 无 caption 评测）。

---

**附**：本报告与 `docs/PickNet_Code_Audit_Report.md`、`docs/PickNet_Code_Provenance.md`、`docs/CapERA_XE_Baseline_Audit.md` 共同构成第三方代码底座档案。PEEK 权重为 CC-BY-NC-SA（非商用），若论文实验使用 `peek_base` 权重需注意许可；自训 scorer 则无此约束（代码 Apache-2.0）。
