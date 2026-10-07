# CapERA Uniform-Frame XE Baseline — xmodaler Pipeline 审计报告

日期：2026-09-29 ｜ 阶段：Step 1–2（审计，未训练）｜ 代码基准：`third_party/xmodaler`（upstream commit `ed1f5590`，Apache-2.0）

本文所有结论均给出 `文件:行号` 级代码证据。行号以 vendored 副本为准。

---

## 1. 审计结论速览

| 问题 | 答案 |
|---|---|
| xmodaler MSVD 用什么视觉 backbone？ | **ResNet-152（pool5 逐帧特征，2048-d）**；官方文档只约定特征目录/格式，**未捆绑任何提取器**，特征离线外部提取 |
| 特征文件格式 | `{video_id}.npy` / `.npz`，内含 `features` 数组 `(T, D)` float32（`functional/func_io.py:21 read_np` 两种都兼容） |
| ATT_FEATS shape | 数据集层：`(T, D)` 单样本；`preprocess_batch` 后：`(B, T_max, D)` |
| ATT_MASKS shape | `(B, T_max)`，float32 0/1（`functional/func_feats.py:11 pad_tensor`） |
| 需要 projection？ | 是：`VisualBaseEmbedding = Linear(IN_DIM→OUT_DIM) + 可选 act/norm/dropout`，**无视觉位置编码** |
| 需要 position encoding？ | 视觉侧：**无**（仅靠帧序 + 注意力）；文本侧：`TOKEN_EMBED.POSITION=SinusoidEncoding` |
| XE loss | `LabelSmoothing`（KL 散度，target≥0 掩码）或 `CrossEntropy(ignore_index=-1)`，作用于 `G_LOGITS (B,L,V)` vs `G_TARGET_IDS (B,L)` |
| 评测 | `COCOEvaler`（pycocotools COCO + pycocoevalcap COCOEvalCap）→ Bleu_1-4 / METEOR / ROUGE_L / CIDEr |
| **硬约束（必须处理）** | `defaults.py:522`：`results.append({ID_KEY: int(id), VALUE: output})` — **`int(id)` 硬编码**，CapERA 字符串 video_id 必须映射为 int 或改这一行 |
| 兼容性 | torch 2.1.2 可跑（`utils/env.py` 仅要求 torch≥1.4）；已补装 fvcore/omegaconf/portalocker/psutil/tabulate/termcolor（纯 Python 小包，无版本冲突） |

---

## 2. 完整数据流（视频 → caption，含文件:行号）

```
CapERA video (mp4, 5s@24fps, 640×640)
  │  (离线，非 xmodaler 内置：需自写提取脚本，见实现计划)
  ▼
features/{split}/{video_id}.npz   →  { 'features': (T, D) float32 }
  │
  ▼ CapERADataset.__call__（新增，仿 msvd.py:76）
  │   read_np → att_feats (T,D)                                    msvd.py:80-82
  │   _sample_frame: T>MAX_FEAT_NUM 时均匀抽 MAX 帧                 msvd.py:70-74
  │   返回 {IDS, ATT_FEATS}，train 另加 G_TOKENS_IDS / G_TARGET_IDS  msvd.py:87-113
  ▼
DefaultTrainer.run_step → preprocess_batch                          defaults.py:531-555
  │   ATT_FEATS → pad → (B, T_max, D)；ATT_MASKS (B, T_max)         base_enc_dec.py:90-98
  │   G_TOKENS_IDS → (B, L_max) + TOKENS_MASKS                      base_enc_dec.py:117-121
  │   G_TARGET_IDS → (B, L_max)，padding=-1                         base_enc_dec.py:123-126
  │   全部 .cuda()                                                  base_enc_dec.py: dict_to_cuda
  ▼
TransformerEncoderDecoder._forward                                   transformer_enc_dec.py:96-126
  │   ① get_extended_attention_mask → EXT_ATT_MASKS (B,1,1,T)、
  │      EXT_G_TOKENS_MASKS (B,1,L,L) 因果掩码（-10000）            transformer_enc_dec.py:53-90
  │   ② VisualBaseEmbedding: Linear(D→H) → ATT_FEATS (B,T,H)        visual_embed.py:78-96
  │   ③ TransformerEncoder: 6×BertLayer 自注意力 → ATT_FEATS (B,T,H) transformer_encoder.py:44-51
  │   ④ TokenBaseEmbedding: Embedding(V,H)+Sinusoid 位置 → G_TOKEN_EMBED (B,L,H)
  │   ⑤ TransformerDecoder: 6×BertGenerationLayer
  │        （自注意力因果 + 与编码器输出交叉注意力）→ G_HIDDEN_STATES [(B,L,H)]×6
  │   ⑥ BasePredictor: Linear(H→V) → G_LOGITS (B,L,V)               base_predictor.py:41-48
  ▼
loss = LabelSmoothing(G_LOGITS, G_TARGET_IDS)                        label_smoothing.py:33-52
  ▼
backward + optimizer.step（梯度裁剪 GRAD_CLIP=0.1）                  defaults.py:562-579
  ▼
推理：model(data, use_beam_search=True, output_sents=True)           defaults.py:515-522
  │   BeamSearcher 逐 token（TIME_STEP + HISTORY_STATES 缓存）       beam_searcher.py:43-141
  │   decode_sequence: 遇 id 0 (EOS) 停止 → 字符串                  func_caption.py:3-14
  ▼
COCOEvaler.eval(results, epoch) → {Bleu_1..4, METEOR, ROUGE_L, CIDEr} coco_evaler.py:35-47
```

维度一致性要求（写 config 时必须满足，否则 shape error）：
`VISUAL_EMBED.OUT_DIM == BERT.HIDDEN_SIZE`、`TOKEN_EMBED.DIM == BERT.HIDDEN_SIZE`、`DECODER_DIM == BERT.HIDDEN_SIZE`（predictor 输入）。MSVD transformer.yaml 全部为 512。

---

## 3. 逐模块细节

### 3.1 Dataset（`xmodaler/datasets/videos/msvd.py`）

- `load_data(cfg)`（:56）：读 anno pkl → `list[{'video_id', 'tokens_ids': (1,L+1), 'target_ids': (1,L+1)}]`；train 时按 caption 展开（每视频 5 句 → 5 条样本）。
- `_sample_frame(atten_feats)`（:70）：`interval = T/MAX_FEAT_NUM; idx = [int(i*interval) for i in range(MAX_FEAT_NUM)]` — **均匀采样，floor 取整**。
- `__call__`（:76）：
  - train：随机取 `SEQ_PER_SAMPLE` 句 caption → 返回 `{IDS, ATT_FEATS(T,D), SEQ_PER_SAMPLE, G_TOKENS_IDS(list of (L,)), G_TARGET_IDS(list of (L,)), G_TOKENS_TYPE(list of ones(L))}`（`dict_as_tensor` 转 torch）。
  - val/test：只返回 `{IDS, ATT_FEATS, G_TOKENS_TYPE: ones(MAX_SEQ_LEN)}`（生成模式，无 caption）。
- 特征路径由 `DATALOADER.FEATS_FOLDER` + `{video_id}.npy` 拼出（:80）。

### 3.2 批处理（`meta_arch/base_enc_dec.py:90-224 preprocess_batch`）

- `pad_tensor(vfeats, padding_value=0, use_mask=True)` → `(B, T_max, D)` + `(B, T_max)` 0/1 掩码（pad 处为 0）。
- 文本同理：`G_TOKENS_IDS → (B, L_max)`（pad=0）+ `TOKENS_MASKS`；`G_TARGET_IDS → (B, L_max)`（pad=-1，loss 端忽略）。
- `SEQ_PER_SAMPLE>1` 时视觉特征按句复制。

### 3.3 注意力掩码（`transformer_enc_dec.py:53-90 get_extended_attention_mask`）

- `EXT_ATT_MASKS`: `(B,1,1,T_max)`，pad 位置 −10000；
- `EXT_G_TOKENS_MASKS`: `(B,1,L,L)` 因果下三角 × TOKENS_MASKS；
- `EXT_U_TOKENS_MASKS`: 双向文本掩码（非 caption 任务不用）。

### 3.4 VisualBaseEmbedding（`embedding/visual_embed.py:30-96`）

`forward`：`ATT_FEATS (B,T,D_in)` → `Linear(D_in→D_out)` → 可选 act(`ACTIVATION`) → 可选 LayerNorm(`USE_NORM`) → 可选 dropout(`DROPOUT`) → 返回 `{ATT_FEATS: (B,T,D_out)}`。
`embeddings_pos` 仅当 `LOCATION_SIZE>0`（图像 region 用），**视频不启用**。视觉侧**没有**位置编码，帧序信息由特征顺序隐式承载（与 xmodaler MSVD 官方协议一致）。

### 3.5 TransformerEncoder（`encoder/transformer_encoder.py:18-52`）

`MODEL.BERT.NUM_HIDDEN_LAYERS` 个 `BertLayer`（layers/bert.py，标准 BERT 自注意力：`hidden_size=MODEL.BERT.HIDDEN_SIZE`、`num_attention_heads=MODEL.BERT.NUM_ATTENTION_HEADS`、`intermediate=INTERMEDIATE_SIZE`）。`mode='v'` 时对 `ATT_FEATS` 用 `EXT_ATT_MASKS` 做时间维自注意力；`mode='t'` 返回空（transformer 无文本编码器）。

### 3.6 TokenBaseEmbedding（`embedding/token_embed.py`）

`Embedding(VOCAB_SIZE, DIM)` + `SinusoidEncoding`（`POSITION` 配置）+ 可选 act/norm/dropout/type。生成时 `TIME_STEP` 存在则逐 token 取位置。**BOS/EOS 共用 id 0**（见 §3.9）。

### 3.7 TransformerDecoder（`decoder/transformer_decoder.py:46-77`）

`MODEL.BERT.NUM_GENERATION_LAYERS` 个 `BertGenerationLayer`：自注意力（因果 `EXT_G_TOKENS_MASKS`）+ 与编码器输出 `ATT_FEATS (B,T,H)` 交叉注意力（`EXT_ATT_MASKS`）。训练时一次前向（教师强制）；生成时逐 token，`HISTORY_STATES` 缓存 KV。

### 3.8 BasePredictor（`predictor/base_predictor.py:41-48`）

取 `G_HIDDEN_STATES[-1]` → dropout → `Linear(DECODER_DIM, VOCAB_SIZE)` → `G_LOGITS (B,L,V)`。

### 3.9 词表与 token 协议（`tools/msvd_preprocess.py`）

- 词表：1-indexed（`wtoi = {w: i+1}`）；**id 0 = BOS 且 = EOS**（输入句 `input_Li[0,0]=0`，目标句 `output_Li[0,seq_len]=0`）。
- `tokens_ids` = `[BOS, w1, ..., wn]`；`target_ids` = `[w1, ..., wn, EOS, -1, -1, ...]`。
- 定长 `max_length+1`，超出截断，不足 -1 填充。
- `vocabulary.txt`：每行一个词（第 i 行 → id i+1），不含 BOS/EOS。`load_vocab`（func_io.py:50）读入并在 index 0 补 `'.'` 占位。
- `VOCAB_SIZE = 词数 + 1`。
- **评测解码**：`decode_sequence`（func_caption.py:3）遇 id 0 停止拼词。`predictor` 输出 argmax id 即词 id。

### 3.10 训练入口与流程

- 入口：`third_party/xmodaler/train_net.py` → `launch(main, num_gpus)`（单卡直接跑 main）→ `setup()`：`get_cfg()` → `add_config`（注册模型相关默认配置）→ `merge_from_file` → `merge_from_list(--opts)` → `default_setup`。
- `DefaultTrainer.__init__`（engine/defaults.py:230-295）：build_model / optimizer / train+val+test loaders / evaluators / losses / lr_scheduler / checkpointer（fvcore，保存 `model_Epoch_x_Iter_y.pth = {model, iteration, optimizer, scheduler}`）/ hooks。
- hooks（defaults.py:306-400）：IterationTimer、LRScheduler、ScheduledSampling（**仅 LSTM 类模型消费 `SS_PROB`，Transformer 无效**）、ModelWeightsManipulating、PeriodicCheckpointer（每 `CHECKPOINT_PERIOD` epoch）、EvalHook×2（val/test，每 `EVAL_PERIOD` epoch，epoch≥`*_EVAL_START` 时执行）、PeriodicWriter（JSONWriter + TensorboardWriter，torch 自带 tensorboard）。
- `run_step`（defaults.py:531-579）：取 batch → `preprocess_batch` → `model(data)` → 各 loss 求和 → `backward` → `optimizer.step()`。
- **梯度裁剪**：由 `SOLVER.GRAD_CLIP=0.1` 经 optimizer builder 生效（默认 Adam）。

### 3.11 推理与评测

- `DefaultTrainer.test`（defaults.py:506-529）：`model.eval()` → 遍历 val/test loader → `model(data, use_beam_search=True, output_sents=True)` → `res[kfg.OUTPUT]` 为字符串列表 → **`results.append({ID_KEY: int(id), VALUE: output})`（:522）**。
- `COCOEvaler`（evaluation/coco_evaler.py:21-47）：
  - `annfile` = COCO 风格 JSON：`{"images": [{"id": int, "file_name": str}], "annotations": [{"image_id": int, "id": int, "caption": str}]}`（由 `msvd_preprocess.save_split_json_file` 生成，每视频 5 条 annotation）。
  - `results` = `[{"image_id": int, "caption": str}]`。
  - 流程：`COCO(annfile)` → 临时文件写 results → `coco.loadRes` → `COCOEvalCap.evaluate()` → 返回含 `Bleu_1/2/3/4、METEOR、ROUGE_L、CIDEr` 的 dict。
  - 注意 `kfg.TEMP_DIR='./data/temp'`（相对运行目录，repo 已自带该空目录；`__init__` 里不存在才 mkdir）。**训练/评测必须在 `third_party/xmodaler` 目录下启动，或改 TEMP_DIR。**
  - `sys.path.append('../coco_caption')` 为遗留路径（环境已装 pycocoevalcap，无影响）。
- `--eval-only`：加载 `MODEL.WEIGHTS` 后对 val/test 各跑一遍 test() 并打印。

### 3.12 环境依赖结论（uavcap 实测）

xmodaler 本体新补装：`tabulate`、`termcolor`、`fvcore`、`omegaconf`、`portalocker`、`psutil`（均为纯 Python 或轻量包，与 torch 2.1.2/numpy 1.26.4 无冲突，已验证 import）。`transformers` 仅 `tokenization_bert.py` 用到旧包名 `pytorch_transformers`（BERT tokenizer 路径，本 baseline 不用）。

---

## 4. Feature extractor 决策（对应需求 §四）

### xmodaler 原协议（方案 A 依据）

- 官方文档（`docs/tutorials/using_builtin_datasets.md`）：MSVD = `features/resnet152/*.npy`；MSR-VTT = `msrvtt_torch/feature/resnet152/*.npy`。
- 配置印证：`configs/video_caption/msvd/*.yaml` 中 `VISUAL_EMBED.IN_DIM=2048`（ResNet-152 pool5 维度）；`MAX_FEAT_NUM` = 25（base）或 50（transformer）。
- **xmodaler 不带提取器**：特征提取必须在框架外完成（社区惯例为 jssprz/video_features 的 resnet152 单帧特征）。

### 方案对比

| 维度 | A: ResNet-152（2048-d，原协议） | B: MaxViT-S（768-d，timm `maxvit_small_tf_224`） |
|---|---|---|
| 与 xmodaler 协议一致 | 完全一致（IN_DIM=2048） | 一致（IN_DIM 可配置为 768，其余协议不变） |
| 与项目模型计划一致（CLAUDE.md） | 否 | **是**（CapERA 论文 backbone，已在 GPU 验证） |
| 官方预训练 captioner 权重可初始化 | 理论可（MSVD/MSR-VTT ckpt）但**权重在 Google Drive，本机不可达** → 实际不可用 | 不兼容官方 ckpt（维度不同） |
| 权重获取 | torchvision 下载（download.pytorch.org，需测速）或 timm | timm → HF Hub（本机学术加速可达） |
| 特征质量（UAV 域） | 2015 年 ImageNet 特征，较弱 | 现代强特征，CapERA 论文验证有效 |
| 训练/存储成本 | 2048-d × T | 768-d × T（小 2.7×） |

### 决策：**主选 B（MaxViT-S，768-d），脚本同时支持 A（`--backbone resnet152`）作对照**

理由：(1) 唯一对 A 有利的论据"复用 xmodaler 官方预训练权重"在本机不可用（权重在 Google Drive，网络不可达），A 的协议优势只剩"与论文报告数字可比"，而我们并不复现 MSVD 数字；(2) B 与项目既定模型计划（MaxViT-S per-frame encoder）一致，后续 selector 阶段特征口径统一，避免二次换 backbone；(3) 两种方案在 xmodaler 侧的兼容性完全等价（只是 `VISUAL_EMBED.IN_DIM` 一个数）；(4) B 特征更强、维度更小、权重获取路径确定（HF，已开通加速）。A 保留为消融选项。

### 帧数 N 决策：**候选池 10 帧（2 fps），训练输入 8 帧（MAX_FEAT_NUM=8）**

- xmodaler 原协议：MSVD 25/50 帧（10–15s 视频）；CapERA 仅 5s@24fps=120 帧，视频更短 → 更少帧合理。
- 采纳建议 N=8：`MAX_FEAT_NUM=8`，全帧数 T=10（2 fps）> 8 → 触发 xmodaler 原生 `_sample_frame` 均匀抽 8 帧——**完全复用框架协议，一行不改**；10 帧候选池同时为第二阶段 selector 提供候选特征（无需重提取）。
- 需要逐帧候选时可随时用同一脚本 `--fps 24` 重提取（Phase 2 再定）。

---

## 5. 必须遵守的兼容性清单（CapERA 适配核对表）

1. **int id 硬约束**：`defaults.py:522 int(id)` → 预处理给每个视频分配全局唯一 int id（train 0..1472、test 1473..2863），保存 `video_id_map.json`（int ↔ 原始字符串，含 split）。**不改 xmodaler**。
2. 特征文件：`features/CapERA/{backbone}/{train|test}/{int_id}.npz`，内含 `features`（T=10, 768）float32。split 隔离解决 train/test video_id 字符串重叠问题。
3. anno pkl：`{video_id: int, tokens_ids: (1,L+1) uint32, target_ids: (1,L+1) int32}`，BOS/EOS 协议同 §3.9。
4. `vocabulary.txt` + `VOCAB_SIZE=len+1`；`MAX_SEQ_LEN` 按 caption 长度统计设定（L 含 BOS/EOS）。
5. eval JSON（COCO 风格）：images/annotations 用同一 int id；每视频 5 条 annotation（保证 reference 一一对应）。
6. 训练/评测脚本 cwd 必须在 `third_party/xmodaler`（`data/temp`、相对路径依赖）；config 内路径用绝对路径。
7. val split：官方 test 不动；从官方 train 划固定 5%（73 视频，seed 固定）作 val 用于早停/调参；train 为剩余 1400 视频。**pkl 按 split 分开生成**。
8. 不修改 `third_party/xmodaler` 任何文件；新增文件仅：`xmodaler/datasets/videos/capera.py`（框架包内新增 module，属加法）、项目内 `scripts/`、`configs/`。

---

## 6. 遗留风险（实现阶段验证）

1. timm MaxViT-S 权重下载（HF Hub，已开通加速，需实测）；
2. pycocotools 对 int id 的 COCO 评测（xmodaler MSVD 同款用法，风险低）；
3. `COCOEvalCap` 需要 `data/temp` 可写；
4. val/test loader 无 shuffle、`drop_last=False`，评测全量覆盖。
