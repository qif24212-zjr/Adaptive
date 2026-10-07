# CapERA Uniform-Frame XE Baseline — 实现计划

日期：2026-09-29 ｜ 依据：[CapERA_XE_Baseline_Audit.md](CapERA_XE_Baseline_Audit.md)

## 0. 目标与边界

- 目标：`CapERA video → uniform 8/10 帧 → MaxViT-S 特征 → xmodaler TransformerEncoder/Decoder → caption`，仅 XE（LabelSmoothing）训练，跑通 train / val / test / CIDEr 评测。
- **不实现**：SCST/RL、PickNet/AdaFrame、任何 selector。Dataset 只提供候选特征（10 帧/视频），selector 属未来模型内部模块。
- **不改 `third_party/xmodaler` 上游任何文件**：CapERADataset 为新模块（框架包内加法），入口脚本在项目侧自写（`scripts/train_capera.py`），通过显式 `import xmodaler.datasets.videos.capera` 注册数据集。

## 1. 数据协议（全部新产物，不碰原始 CapERA）

```
data/CapERA/                          # 派生标注（新目录）
├── video_id_map.json                 # {"train": {"Baseball_001.mp4": 0, ...}, "test": {...}}（int id 全局唯一，train 0..1472, test 1473..2863）
├── vocabulary.txt                    # 每行一词，id=i+1；id 0 = BOS/EOS
├── capera_caption_anno_train.pkl     # [{'video_id': int, 'tokens_ids': (1,L+1) uint32, 'target_ids': (1,L+1) int32}]
├── capera_caption_anno_val.pkl       # train 内固定 5%（74 视频，seed 固定）
├── capera_caption_anno_test.pkl
├── captions_val.json                 # COCO 风格 {images:[{id}], annotations:[{image_id,id,caption}]}（5 句/视频）
└── captions_test.json

features/CapERA/maxvit_s/             # 特征（2 fps → 每视频 10 帧）
├── train/{int_id}.npz                # {'features': (10, 768) float32}
└── test/{int_id}.npz
```

- 词表：pycocoevalcap 自带 PTBTokenizer 分词（与评测端 COCOEvalCap 同 tokenizer），词频阈值 1（全保留，无 UNK）。
- `L = 最长 caption token 数 + 2`（BOS/EOS），超长截断（99 分位封顶，防异常值）。
- split 身份：val 从官方 train 抽取（固定 seed），官方 test 完整保留 1391 视频。

## 2. 新增文件清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/preprocess_capera.py` | JSON → int-id 映射 / 词表 / pkl / COCO 风格 eval JSON | 新 |
| `scripts/extract_capera_feats.py` | mp4 → 均匀 10 帧 → MaxViT-S(768) 或 ResNet-152(2048) → npz | 新 |
| `third_party/xmodaler/xmodaler/datasets/videos/capera.py` | `CapERADataset`（仿 msvd.py，split 子目录加载特征） | 新（加法） |
| `configs/capera/xe_baseline.yaml` | XE 训练/评测配置（路径绝对化） | 新 |
| `scripts/train_capera.py` | 项目侧训练入口（注册 capera 后复用 xmodaler DefaultTrainer） | 新 |
| `scripts/train_capera_xe.sh` | 训练启动脚本（cwd=third_party/xmodaler） | 新 |
| `scripts/eval_capera.sh` | --eval-only 评测脚本 | 新 |
| `scripts/smoke_test_capera.py` | 4 项冒烟测试 | 新 |

## 3. CapERADataset 设计（capera.py）

- `from_config`：anno 三 pkl（绝对路径）+ `MAX_FEAT_NUM=8` + `FEATS_FOLDER=features/CapERA/maxvit_s`；stage→子目录映射 `train/val → 'train'`、`test → 'test'`。
- `__call__`：与 `msvd.py:76-113` 同构：`read_np → (T,768) → T>8 时 _sample_frame 均匀抽 8 → {IDS(int), ATT_FEATS}`；train 加 `G_TOKENS_IDS/G_TARGET_IDS/G_TOKENS_TYPE/SEQ_PER_SAMPLE`；val/test 只返回特征（生成模式）。
- 数据流保持：`Dataset → preprocess_batch → ATT_FEATS (B,8,768) / ATT_MASKS (B,8) → visual_embed(768→512) → encoder → decoder → predictor`。

## 4. 模型配置要点（xe_baseline.yaml）

- 骨架照抄 `configs/video_caption/msvd/transformer/transformer.yaml`（6 层编码器 + 6 层解码器、512 隐层、8 头、intermediate 2048），仅改：`VISUAL_EMBED.IN_DIM 768`、`MAX_FEAT_NUM 8`、`VOCAB_SIZE`、`MAX_SEQ_LEN`、路径、`SOLVER.EPOCH 50 / BASE_LR 1e-4 / CHECKPOINT_PERIOD 5 / EVAL_PERIOD 5`、`SEED 42`、`OUTPUT_DIR experiments/capera_xe_baseline`。
- 训练时长预估：7365 样本、batch 64、~115 step/epoch，4090 上每 step 约 0.1–0.2s → 50 epochs ≈ 15–25 分钟（不含评测）。

## 5. 冒烟测试（scripts/smoke_test_capera.py，全部通过才开全量训练）

- **Test 1（dataset，2–4 视频）**：train/test 各取 2 样本走 `CapERADataset.__call__` → 断言 `ATT_FEATS (8,768)`、token 协议（首 token 0=BOS、target 末位 0=EOS、-1 填充）、`G_TOKENS_TYPE` 全 1。
- **Test 2（forward）**：`build_model(cfg)`，batch=2 走 `preprocess_batch + model(data)` → 断言 `G_LOGITS (B,L,V)`、无 NaN；再跑一次 greedy 生成出 2 句 caption。
- **Test 3（backward）**：LabelSmoothing loss → backward → 断言 loss finite、梯度无 NaN、`optimizer.step()` 后参数变化。
- **Test 4（overfit，4 视频 × 5 句）**：小 datalist 训练 ~200 step，断言末段均损 < 首段均损 × 0.7 且持续下降。
- 全部测试脚本自包含（不污染正式 anno/checkpoints 目录）。

## 6. 执行顺序

1. `preprocess_capera.py`（CPU，秒级）→ 校验：id 映射、词表、pkl、JSON、5 句/视频。
2. `extract_capera_feats.py --backbone maxvit_s --fps 2`（GPU，约 3–10 分钟）→ 校验：2864 个 npz、(10,768) shape、无 NaN。
3. smoke test 1–4 全绿。
4. `train_capera_xe.sh`（后台）→ 完成后 `eval_capera.sh --eval-only`（MODEL.WEIGHTS=model_final.pth）出 BLEU-4/METEOR/ROUGE-L/CIDEr。
5. 报告：训练曲线（metrics.json）、测试集指标、每 epoch val CIDEr 走向。

## 7. 风险与预案

- timm MaxViT-S 权重下载失败 → 换 `--backbone resnet152`（同脚本支持）或检查网络加速；
- COCO 评测 id 必须 int → 已由 int-id 映射解决（不改 xmodaler 的 `int(id)`）；
- 训练/评测 cwd 必须为 `third_party/xmodaler`（`data/temp`、相对路径）→ shell 脚本内固定 `cd`；
- 若 overfit 测试 loss 不降 → 先查特征文件（全 0/NaN）、token 协议、lr；不盲目开全量。

---

## 8. 执行结果（2026-09-29 完成）

- 预处理：train 6995 / val 74 / test 1391 caption entries；vocab 1890 词（VOCAB_SIZE 1891）；MAX_SEQ_LEN 26；int-id 映射 2864 视频（train 0..1472, test 1473..2863）。
- 特征：`features/CapERA/maxvit_small_tf_224/{train,test}/{id}.npz`，2864/2864，全部 (10, 768) float32，0 失败（15 分钟）。
- 冒烟测试 1–4 全部通过（dataset / forward+greedy+beam / backward / 4 视频 overfit loss 4.06→0.87）。
- 全量 XE 训练（50 epochs, batch 64, lr 1e-4, WarmupLinear, LabelSmoothing 0.1, seed 42）：约 20 分钟；checkpoints 在 `experiments/capera_xe_baseline/`。
- **最终指标（beam 5）**：

| split | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| val（74，取自 train，仅作监控/早停） | 0.336 | 0.263 | 0.507 | 1.276 |
| **test（1391，官方 split）** | **0.170** | **0.185** | **0.388** | **0.610** |

- 训练期 val 曲线：CIDEr epoch5=0.997 → 峰值 epoch35≈1.36 → epoch50=1.276（轻微过拟合，符合小数据集 XE 基线预期）。
- 预测文件：`experiments/capera_xe_baseline/results/50.json`（1391 条，0 空句）。

### 途中修复的问题（全部有据可查，见 logs/ + THIRD_PARTY_PATCHES.md）

1. `pytorch_transformers` 旧包名（xmodaler 1 行补丁）；
2. 缺依赖：tabulate/termcolor/fvcore/omegaconf/portalocker/psutil/json-lines/jsonlines/setuptools<81；
3. SPICE 下载卡死 → 入口脚本 no-op 替换（只报告 BLEU-4/METEOR/ROUGE-L/CIDEr）；
4. val/test pkl 每视频 5 条 entry → 评测重复预测崩溃 → 改为 1 条/视频；
5. 早期模型空 caption → BLEU 断言崩溃 → CapEraTrainer 占位符兜底；
6. greedy decode 原地改写输入 dict → 冒烟测试 beam 前重建 batch（上游行为，非 bug）。
