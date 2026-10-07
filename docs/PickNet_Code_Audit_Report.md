# PickNet / Video Captioning 代码调研审计报告

日期：2026-09-29 ｜ 状态：**仅调研，未训练、未修改任何数据、未动核心目录**（本次仅新增 `docs/` 与 `/root/autodl-tmp/repo_audit/` 只读克隆）

---

## 0. 结论先行（对应最终要求 Q1–Q7）

### Q1. 是否找到可以直接复用的 PickNet 实现？

**NO。**

- **无官方代码**：ECCV 2018 官方页面、arXiv 1803.01457（含 ar5iv 全文）、Papers with Code、HuggingFace Papers 均无任何代码链接；论文正文自己写道 "most of state-of-the-art methods do not release executable codes"。
- **无可信第三方实现**：GitHub 仓库/话题/README 全检索（约 20 组查询）、Gitee、grep.app / Sourcegraph / searchcode 代码级搜索、DuckDuckGo/Bing/多轮 WebSearch、中文社区（知乎/CSDN）均无 PickNet（视频描述版）复现。GitHub 上所有名为 PickNet 的仓库都是**地震学震相拾取**，与本论文无关。
- 结论：不存在"现成 PickNet 代码"可复用的可能，只能**在成熟 Video Captioning 框架上重建 PickNet 式 selector**（工作量可控，见 §7/§14）。

### Q2. 如果 YES 哪个 repo？（N/A）

不适用（Q1 = NO）。

### Q3. 有没有成熟的 Video Captioning framework 可以作为替代基础？

**YES —— 推荐 [YehLi/xmodaler](https://github.com/YehLi/xmodaler)（Apache-2.0，X-LAN/M2-Transformer 作者团队官方框架）。**

核心理由：
1. **依赖兼容**：Python ≥3.6、PyTorch ≥1.8 → 直接兼容当前 uavcap 环境（3.10.13 / 2.1.2+cu121 / RTX 4090），无需降级；
2. **训练机制完整**：既有 XE（LabelSmoothing + scheduled sampling），**又有现成的 SCST 强化学习 trainer**（`rl_trainer.py` 实现 sample 解码 − greedy 基线 = advantage，`reward_criterion.py` 实现 REINFORCE loss）——这正是 PickNet 式 selector RL 训练可以直接复用的机制；
3. **评测完整**：`COCOEvaler`（pycocotools + pycocoevalcap → BLEU/METEOR/ROUGE-L/CIDEr，本环境已装依赖），PickNet 论文用的就是同源 coco-caption 指标；
4. **注册表架构**：datasets / models / losses / engines 全部 registry 注册，**扩展自定义数据集与自定义模块无需 fork 框架**；
5. **插入点天然存在**：数据集层 `MSVDDataset._sample_frame()` 就是"均匀选帧基线"，adaptive selector 的替换位置明确（§14）。

备选（均不推荐作基础）：
| Repo | 状态 | 问题 |
|---|---|---|
| [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) | 经典 SCST（COCO 图像描述） | torch 0.4 时代，视频版需 fork 大改 |
| [nasib-ullah/video-captioning-models-in-Pytorch](https://github.com/nasib-ullah/video-captioning-models-in-Pytorch) | 经典模型合集（S2VT/SA-LSTM/MARN/RecNet/mean-pooling，**无 PickNet**） | 锁死 torch==1.0.1、numpy==1.16，无法在当前环境运行；仅可当算法参考 |
| [xiadingZ/video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch) | S2VT+Attention，MSR-VTT | torch 0.4，未实现 RL |
| [jayleicn/recurrent-transformer](https://github.com/jayleicn/recurrent-transformer)（MART） | ACL 2020 段落描述 | **Python 2.7**，段落级任务不匹配 |
| [momentslab/peek](https://github.com/momentslab/peek) | **BMVC 2026 官方 selector-only**（蒸馏+ListMLE，非 RL） | 只有选帧器，**下游 captioning 评测代码未发布**；可作为相关文献 + 可选的非 RL 对比 baseline，其 selector 可单独复用 |

### Q4. PickNet 当前的 frame selection 是否已经足够 adaptive？

**部分 adaptive，但缺三样关键东西。**

- **已经 adaptive 的**：每视频选帧数 `N_p` 不是固定 K —— 顺序扫描候选帧逐帧做伯努利 pick/drop 决策，`N_p ∈ [N_min=3, N_max]`（N_max 训练中从 1/3 帧数收缩到 τ=7），平均 6–8 帧。简单/复杂视频理论上会得到不同帧数。
- **缺 ①**：**没有 stop/lookahead 机制**。必须把整个候选序列扫完（每帧都要算 glance 并做决策），无法"看到一半觉得够了就提前停"；没有效用（utility/value）估计，也就没有"预算–质量"折中旋钮。
- **缺 ②**：**没有全局记忆（global memory）**。状态只有"上一个被选帧的灰度 glance 差"和 captioner 隐态，缺乏整个视频的低成本全局上下文来指导"该看哪里"。
- **缺 ③**：**reward 不是逐帧信息增益**。语言 reward 是整句 CIDEr，视觉 reward 是已选特征的标准差——都没有"当前决策为 caption 贡献了多少边际信息"的概念；且按论文机制每步决策都要跑一遍 captioner 生成句子算 CIDEr，训练开销极大（这是 2018 年论文没有成为主流的原因之一）。

### Q5. AdaFrame 的哪些机制最适合借鉴？（只列机制）

1. **Global memory**：低分辨率（112×112）+ 时间下采样（16 帧）的轻量特征 + 位置编码，用注意力按当前隐态查询 → 给 selector 提供"全视频概览"，开销极小；
2. **Utility network（价值函数/critic）**：一个 FC 层从隐态回归"预期未来收益"，用 rollout 回报做回归训练——用于**预测"再看几帧还能涨多少"**；
3. **Adaptive lookahead inference**：测试时维护 running max utility，`V̂_max > V̂_t + μ` 连续 `p` 步则停止（patience）；**μ 是精度-计算量折中的连续旋钮**，且停止判据基于每个样本自身的 utility 分布；
4. **selection 与 stop 解耦**：论文实验证明"二值 stop 策略与选帧策略联合 RL 训练"很难收敛，用 utility 做停止信号更稳；
5. （可选）连续位置策略（可前跳/回看）——对 captioning 未必必要，顺序扫描 + pick/drop 已够用。

不复制 AdaFrame 的：分类置信度 reward（captioning 没有分类信号）、LSTM 主体结构（我们已有 Transformer captioner）。

### Q6. 我的创新模块最合理的插入位置？

以 xmodaler 真实代码结构为准（文件级证据见 §5/§6）：

```
CapERADataset (xmodaler/datasets/videos/capera.py, 需新增, 仿 msvd.py)
   ↓ 全帧特征 {video_id}.npz (T×D)          ← 现代码在此处做均匀采样 _sample_frame()
[Adaptive Selection Module]                  ← ① 我的方法：FrameSelector
   │   GlobalMemory(AdaFrame式) + Policy(pick/drop或预算分配)
   │   + UtilityHead(caption导向预期收益) + Stop(lookahead)
   ↓ 被选帧子集 (k≤T, per-video 自适应)
preprocess_batch (base_enc_dec.py) → ATT_FEATS/ATT_MASKS
   ↓
visual_embed → TransformerEncoder → TransformerDecoder → BasePredictor
   ↓                                    (xmodaler 原装 captioner，不重写)
loss: LabelSmoothing (XE阶段) / RewardCriterion (RL阶段, 复用 rl_trainer.py + reward_criterion.py)
inference: greedy/beam → COCOEvaler → BLEU/METEOR/ROUGE-L/CIDEr
```

即：**selector 插在"候选帧特征 → 编码器"之间**，captioner 完全复用 xmodaler；RL 训练复用其 SCST 骨架（阶段 2：冻结 captioner 训练 selector；阶段 3：联合微调），与 PickNet 三阶段训练方案逐条对应（§8）。

### Q7. 最终实验代码建议采用什么结构？

**Vendor xmodaler 到 `third_party/`，只做"加法"，不动其原有文件**：

```
uav_adaptive_captioning/
├── datasets/                 # 数据（只读，不动）
│   ├── CapERA/               # annotations/ + videos/（官方原样）
│   └── WebUAV-3M/            # 后续
├── third_party/xmodaler/     # vendored 框架（上游原样 + 新增下列文件）
│   └── xmodaler/
│       ├── datasets/videos/capera.py        # 新增：CapERADataset（仿 msvd.py）
│       ├── datasets/videos/webuav.py        # 后续新增
│       └── modeling/selector/               # 新增：adaptive selector 模块
│           ├── policy.py / global_memory.py / utility.py / stop.py
│           └── meta_arch/selector_enc_dec.py  # 新增：包一层 meta_arch
├── configs/                  # 实验配置
│   ├── capera/xe_baseline.yaml              # 均匀帧 XE 基线（现成组件拼装）
│   └── capera/selector_rl.yaml              # selector RL 配置
├── scripts/                  # 数据管线（只新增，不改数据）
│   ├── preprocess_capera.py                 # JSON → pkl anno + vocabulary.txt（仿 tools/msvd_preprocess.py）
│   ├── extract_capera_feats.py              # mp4 → MaxViT-S 逐帧 npz
│   ├── train_xe.sh / train_rl.sh / eval.sh
├── experiments/ logs/ checkpoints/ features/  # 已有空目录，沿用
└── docs/                     # 本报告等
```

原则：**框架一行不改，扩展走 registry；换数据集只换 dataset wrapper；换 selector 只换 selector 模块**。这同时天然满足 CapERA ↔ WebUAV-3M 互换（§12/§13）。

---

## 1. Repository 调研汇总

**PickNet（ECCV 2018）与 AdaFrame（CVPR 2019）：均无官方代码，也无可信第三方复现。** 证据链：

| 检索途径 | PickNet | AdaFrame |
|---|---|---|
| 官方论文页（openaccess.thecvf.com） | 无代码链接 | 无代码链接（仅 supplementary PDF） |
| arXiv 摘要页 + ar5iv 全文 | 无链接（全文仅引用 jcjohnson/cnn-benchmarks、tylin/coco-caption 等工具库） | 无链接 |
| Papers with Code | 无实现注册 | "No code implementations yet" |
| HuggingFace Papers API（1803.01457 / 1811.12432） | implementations=0 | implementations=0 |
| GitHub 仓库搜索（API 约 20 组查询：picknet / picknet+video / frame+picking / 1803.01457 / topic:video-captioning 等） | 全部命中为地震学 PickNet | 全部命中为无关项目 |
| GitHub 用户搜索（一作 Zuxuan Wu = blackfeather-wang 全部仓库；二作 Chih-Yao Ma = chihyaoma 全部仓库） | — | **一作 GitHub 无 AdaFrame / FrameGlance 仓库** |
| 候选 video-captioning 仓库 README 全文 grep（9 个） | 0 命中 | — |
| Gitee（API + 页面） | 0 命中 | 0 命中 |
| 代码级搜索 grep.app / Sourcegraph / searchcode | 被 WAF 拦截或 0 命中 | 同左 |

**候选清单（本次实际克隆并读码的）**：

| # | Repo | 官方? | 对应论文 | 星/维护 | 结论 |
|---|---|---|---|---|---|
| 1 | [YehLi/xmodaler](https://github.com/YehLi/xmodaler) | 是（X-LAN 作者） | X-LAN / M2-Transformer 等（CVPR 2020 等） | 971★，master 最新提交 2023-02，Apache-2.0 | **✅ 推荐基础** |
| 2 | [momentslab/peek](https://github.com/momentslab/peek) | 是 | PEEK（BMVC 2026, arXiv:2605.31029） | 20★，2026-05 新建，Apache-2.0 | 相关文献 + 可选非 RL 对比 |
| 3 | [nasib-ullah/video-captioning-models-in-Pytorch](https://github.com/nasib-ullah/video-captioning-models-in-Pytorch) | 否（合集） | S2VT/SA-LSTM/MARN/RecNet | 73★ | 仅算法参考，依赖过老 |
| 4 | [jayleicn/recurrent-transformer](https://github.com/jayleicn/recurrent-transformer) | 是 | MART（ACL 2020） | 170★ | Python 2.7，任务不符 |
| 5 | [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) | 是 | Self-Critical（CVPR 2017） | 高 | SCST 机制参考，torch 0.4 |
| 6 | [xiadingZ/video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch) | 否 | S2VT+Att（MSR-VTT） | 399★ | torch 0.4，无 RL |
| 7 | [syyeung/frameglimpses](https://github.com/syyeung/frameglimpses) | 是 | FrameGlimpse（CVPR 2018） | 94★ | AdaFrame 前身参考（location+stop 分支），旧 torch |

## 2. Original Paper

- **PickNet**: Yangyu Chen, Shuhui Wang, Weigang Zhang, Qingming Huang. "Less Is More: Picking Informative Frames for Video Captioning." ECCV 2018. [开放获取页](https://openaccess.thecvf.com/content_ECCV_2018/html/Yangyu_Chen_Less_is_More_ECCV_2018_paper.html) / [arXiv:1803.01457](https://arxiv.org/abs/1803.01457)
- **AdaFrame**: Zuxuan Wu, Caiming Xiong, Chih-Yao Ma, Richard Socher, Larry S. Davis. "AdaFrame: Adaptive Frame Selection for Fast Video Recognition." CVPR 2019. [开放获取页](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html) / [arXiv:1811.12432](https://arxiv.org/abs/1811.12432)

## 3. Code authenticity

- PickNet / AdaFrame：无任何公开代码 → authenticity 无从谈起（这就是 Q1=NO 的完整含义）。
- xmodaler：X-LAN（Yehao Li 等）官方框架，971★，Apache-2.0（JD 版权），与论文对得上（内置 X-LAN / M2-Transformer / GCN-LSTM 等 + MSVD/MSR-VTT 复现配置与预训练模型）。**真实可用，且在本机已实际读码验证**。
- peek：BMVC 2026 官方代码 + HF 权重（CC BY-NC-SA），Apache-2.0；selector 训练/推理完整，下游 captioning 评测代码明确标注"still being prepared"。

## 4. Dependency compatibility（以当前环境 Python 3.10.13 / torch 2.1.2+cu121 / RTX 4090 / numpy 1.26.4 为准）

| Repo | 依赖要求 | 兼容性 |
|---|---|---|
| **xmodaler** | Python ≥3.6，PyTorch ≥1.8，pycocotools，pycocoevalcap | ✅ **兼容**。唯一注意：`xmodaler/tokenization/tokenization_bert.py` 内 2 处 `from pytorch_transformers...`（旧包名），仅在启用 BERT tokenizer 时才触达；可用 `transformers`（已装 4.36.2）做 2 行等价替换，不影响训练主路径 |
| nasib-ullah | torch==1.0.1.post2、numpy==1.16.2、h5py==2.9 | ❌ 与 torch 2.1/NumPy ABI 冲突，只能看代码 |
| MART | Python 2.7 | ❌ |
| self-critical.pytorch | torch 0.4 时代 API（Variable、volatile 等） | ❌ 需大改 |
| video-caption.pytorch | torch 0.4 | ❌ |
| peek | 现代（transformers/timm/mobileclip） | ✅ 可跑（如作为对比 baseline） |
| frameglimpses | 旧 torch | ❌ 仅参考 |

## 5. Model architecture（xmodaler，实测读码）

文件 → 结构映射（`third_party/xmodaler/`，clone commit `ed1f5590`）：

| 组件 | 文件 | 说明 |
|---|---|---|
| meta 架构 | `xmodaler/modeling/meta_arch/transformer_enc_dec.py` | `TransformerEncoderDecoder`：`_forward()` = visual_embed → encoder → decoder；`get_extended_attention_mask()` 构造视觉/文本/因果掩码 |
| 基类 | `xmodaler/modeling/meta_arch/base_enc_dec.py` | `preprocess_batch()`（pad+mask）、`greedy_decode()`、`decode_beam_search()` |
| 视觉嵌入 | `xmodaler/modeling/embedding/`（VisualBaseEmbedding，`IN_DIM→OUT_DIM`） | 帧特征投影；**IN_DIM 配置化 → MaxViT-S(768)/CLIP/ResNet 特征直接换** |
| 编码器 | `xmodaler/modeling/encoder/transformer_encoder.py` | 时间维自注意力（M2 风格） |
| 解码器 | `xmodaler/modeling/decoder/transformer_decoder.py` | 自回归 Transformer（BERT 式，**非 GPT-2**；cap 生成层数/头数可配） |
| 预测头 | `xmodaler/modeling/predictor/base_predictor.py` | 隐态 → 词表 logits |
| 词表 | 自建 word-level（`tools/msvd_preprocess.py`，按词频阈值构建，含 `<BOS>/<EOS>`） | CapERA 同样流程 |
| 损失 | `xmodaler/losses/cross_entropy.py`、`label_smoothing.py`（0.1） | XE 阶段 |
| RL 损失 | `xmodaler/losses/reward_criterion.py` | `loss = mean(−logP · rewards)`，EOS 前掩码，标准 REINFORCE |
| 推理 | `DECODE_STRATEGY: BeamSearcher, BEAM_SIZE=5`（或 greedy） | 生成 caption |

注意：CLAUDE.md 模型计划中的 "GPT-2 风格解码器" 与 xmodaler 原装 TransformerDecoder 不是同一个东西。两个选项：**(a) 基线先用原装 TransformerDecoder**（推荐，零新代码，架构族相同：时间自注意力编码 + 自回归 Transformer 解码）；**(b) 之后如需对齐 CapERA 论文基线**，写一个 `GPT2Decoder` 注册进 decoder registry（GPT-2 权重可用 transformers 4.36.2 的 `GPT2LMHeadModel`，工作量一个模块）。先 (a) 后 (b)。

## 6. Frame selection mechanism（对照三方法，全部依据论文原文，非猜测）

### 6.1 PickNet（ECCV 2018，论文原文提取）

- **候选帧来源**：视频逐帧视觉特征（CNN 帧级特征）；同时每帧生成 56×56 灰度缩略图（"glance"）。
- **Selector**：**2 层前馈网络**（非 RNN！），2 输出 = pick/drop 概率（伯努利分布）。输入 = 当前帧 glance − 上一个被选帧 glance（灰度差图，展平 1D）。首帧强制选中。
- **状态/记忆**：无显式记忆；状态 = 上一被选帧 glance + Encoder-Decoder 的隐态（每次 pick 后更新）。
- **决策流程**：按时间顺序扫描全部候选帧，逐帧采样 pick/drop；**扫完整个视频为止（无提前停止机制）**。
- **帧数**：per-video 自适应，`N_p ∈ [N_min=3, N_max]`，N_max 初值 = 总帧数 1/3、训练中收缩至 τ=7；实测平均 6–8 帧。
- **Reward**：`r(v_i) = λ_l·r_l + λ_v·r_v`（若 `N_min ≤ N_p ≤ N_max`，否则罚 `R⁻=−1`）；`r_l` = 用当前选中帧生成的 caption 的 **CIDEr**；`r_v` = 已选帧特征的**标准差**（视觉多样性）。
- **训练**：3 阶段 —— ①监督阶段：全帧训练 Encoder-Decoder（XE + schedule sampling，SGD）；②强化阶段：**冻结** Encoder-Decoder 当环境，REINFORCE（self-critical 基线 `r(â)`）训练 PickNet；③适应阶段：近似联合训练（前向产生 pick，把 pick 当固定选择训 Encoder-Decoder，PickNet 走 REINFORCE）。
- **编码-解码器**：LSTM 编码 + GRU 解码；嵌入 512、隐态 1024；dropout 0.5；Adam；MSVD batch 128 / MSR-VTT 256；每阶段 ≤100 epochs。
- **工程代价警示（实现层面）**：论文机制下每个 pick 决策后都要跑 captioner 生成句子算 CIDEr → 极贵。落地时通常改为"每视频采样一个子集 → 生成一次 → 子集级 REINFORCE"（PickNet 论文的 adaptation 阶段实际上也是近似联合训练）。

### 6.2 AdaFrame（CVPR 2019，论文原文提取）

- **Agent**：memory-augmented **LSTM**，输入 = concat(当前帧特征 v_t, 全局上下文 u_t) + 上一隐态 h_{t-1}/c_{t-1}。
- **Global memory**：16 帧 112×112 低分辨率特征（轻量 MobileNetv2）+ 位置编码；用 h_t 做软注意力查询 → 上下文向量 u_t。
- **Selection network**：高斯位置策略 `ℓ_{t+1} ~ N(sigmoid(W_s·h_t), 0.1²)`，clamp [0,1] → 帧索引；**可前跳/回看**，训练时采样、测试时取均值。
- **Utility network**：1 个 FC 层 `V̂_t = W_u·h_t`，作为 critic 回归折扣未来回报（γ=0.9）——**"再看更多帧的预期收益"估计器**。
- **Reward**：分类置信度的信息增益（预测随新观测变得更自信）；训练固定 K 步 policy gradient。
- **测试（Adaptive Lookahead Inference）**：维护 running max `V̂_max`；当 `V̂_max > V̂_t + μ` 连续 p 步（patience）→ 停止并用当前预测。**μ = 精度↔计算量连续折中旋钮**；停止判据基于每样本自身 utility 分布。
- **关键设计经验**：selection 与 stop **解耦**训练（论文实验证明联合二值 stop 策略难收敛）。

### 6.3 对比表

| Dimension | PickNet | AdaFrame | 我的可能方法 |
|---|---|---|---|
| Task | Video Captioning | Video Recognition | Video Captioning |
| Frame selection | 顺序扫描，逐帧伯努利 pick/drop（2 层 FFN），首帧必选 | 高斯位置策略（连续位置→帧索引，可跳前/回看） | 候选帧上 pick/drop 或预算分配（待设计） |
| Selection state | 上一选中帧的 56×56 灰度差图 + captioner 隐态 | LSTM 隐态 + 当前帧特征 + 全局上下文 u_t | 已选集摘要 + 帧特征（待设计） |
| Memory | 无显式记忆（隐式=编码器隐态） | **Global memory**（16 帧低分辨率特征+注意力） | 借鉴 Global memory + caption 语义记忆（创新点候选） |
| Caption information | 语言 reward = 整句 CIDEr | N/A（无 caption 信号） | **逐帧 caption 边际贡献/信息增益估计（核心创新空间）** |
| Number of frames | 自适应：N_p∈[3, N_max]，平均 6–8 | 训练固定 K；测试自适应（FCVID 8.21 / ActivityNet 8.65 帧平均） | Adaptive per-video（预算/贡献驱动，待设计） |
| Stop mechanism | **无**（必须扫完全部候选帧） | **Utility + running max + μ + patience p 的 lookahead 停止** | 借鉴 lookahead，utility 改为 caption 导向（待设计） |
| Training | 3 阶段（监督→冻结 captioner 训 selector→联合） | 单阶段 policy gradient（固定 K 步）+ utility 回归 loss | 3 阶段沿用 PickNet 结构 + utility 回归（待设计） |
| Reward/objective | λ_l·CIDEr + λ_v·std(已选特征) + 越界罚 −1 | 分类置信度信息增益 | caption 质量 + 视觉多样性 + 预算惩罚 + utility 校准（待设计） |

## 7. Captioning mechanism（xmodaler 实测数据流）

```
video (mp4)
  ↓ 离线: 特征提取 → {video_id}.npz  (T×D，与训练解耦)
CapERADataset.__call__      xmodaler/datasets/videos/capera.py（新增，仿 msvd.py）
  ↓  现代码: _sample_frame() 均匀抽 MAX_FEAT_NUM 帧  ←【固定预算 selector 替换点】
ATT_FEATS (+ATT_MASKS) → preprocess_batch()  pad/掩码   base_enc_dec.py
  ↓
VisualBaseEmbedding (IN_DIM→OUT_DIM)
  ↓
TransformerEncoder（时间自注意力）
  ↓
TransformerDecoder（自回归，G_TOKENS_IDS 教师强制 / 采样）
  ↓
BasePredictor → 词表 logits
  ↓
损失: CrossEntropy / LabelSmoothing（XE 阶段）
     RewardCriterion = mean(−logP·rewards)（RL 阶段: rl_trainer.py 中 sample 解码 − greedy 基线 = advantage）
推理: greedy_decode / decode_beam_search → COCOEvaler → BLEU-4/METEOR/ROUGE-L/CIDEr
```

词表：`tools/msvd_preprocess.py` 从 captions JSON 建词频词表（word-level，`<BOS>/<EOS>`），输出 train/val/test pkl + `vocabulary.txt`。CapERA 每视频 5 句 captions，与 MSVD 同构。

## 8. Training pipeline

- **入口**：`tools/train_net.py`（`--config-file`，detectron2 风格配置合并）；`ENGINE.NAME` 选择 trainer。
- **XE 训练**：`xmodaler/engine/defaults.py` DefaultTrainer（DataLoader 迭代 → `preprocess_batch` → `_forward` → XE loss → 梯度裁剪 0.1 → Adam/AdamW + StepLR/WarmupLinear；每 EVAL_PERIOD 跑 `COCOEvaler`）。
- **RL 训练**：`xmodaler/engine/rl_trainer.py`（另有 `rl_beam_trainer.py` / `rl_mean_trainer.py` 变体）：eval 模式 greedy 解码得基线 reward → train 模式采样解码得 reward → advantage → `RewardCriterion`。**该骨架直接可复用于 selector 的 REINFORCE**。
- **PickNet 三阶段 → xmodaler 落地映射**：
  - 阶段① 监督：`DefaultTrainer` + 全帧（或均匀 MAX_FEAT_NUM 帧）训练 captioner —— 零新代码；
  - 阶段② 强化：冻结 captioner，新写 `SelectorTrainer`（仿 RLTrainer：selector 采样子集 → captioner 生成 → CIDEr+diversity+预算 reward − 均匀子集基线 → REINFORCE）—— 复用 scorer（`xmodaler/scorer/cider.py`）与 RewardCriterion 模式；
  - 阶段③ 适应：联合微调（selector 前向固定 pick 训 captioner + selector 继续 REINFORCE）—— 仿 PickNet 近似联合训练。
- **checkpoint**：`xmodaler/checkpoint/` 保存/加载模型+优化器+iter；EMA 可选。
- **预训练需求**：MSVD/MSR-VTT 有官方 Google Drive 预训练模型（非必需，但可作初始化参考）；CapERA 无现成 checkpoint，需自训。

## 9. Evaluation pipeline

- `xmodaler/evaluation/coco_evaler.py`：`pycocotools.COCO` + `pycocoevalcap.eval.COCOEvalCap`（本环境已装）→ BLEU-4 / METEOR / ROUGE-L / CIDEr。
- 输入：模型在 val/test 上 beam 生成的 `{video_id: caption}` + `INFERENCE.VAL_ANNFILE / TEST_ANNFILE`（`{video_id: [5 参考句]}` 格式 JSON，与 CapERA 完全同构）。
- 额外可加：解码多样性/重复率（参考 MART 仓库的 diversity 脚本思路，非必须）。

## 10. Dataset support

- xmodaler 内置：MSCOCO / MSVD / MSR-VTT / VATEX / TVCaption 等 wrapper（`xmodaler/datasets/{images,videos}/`），`DATASETS.TRAIN/VAL/TEST` 注册表切换。
- 扩展指南见 `xmodaler/datasets/README.md`（官方文档明示如何接自定义数据集：anno pkl + vocabulary.txt + 特征目录 + captions json）。
- CapERA（本机已验证）：train 1473 / test 1391 视频，每视频 5 句英文 caption，`{video_id, annotation:{English_caption:[5]}}` 结构；视频在 `datasets/CapERA/videos/Videos/{Train,Test}/`（官方 ERA 包原样，含 ERA_Dataset.zip 原件未动）。**注意：id 字符串在 train/test 间有重叠，必须按 (split, video_id) 键控或分目录存特征。**

## 11. Checkpoint availability

- xmodaler：官方 Model Zoo（COCO + MSVD + MSR-VTT 预训练模型，Google Drive 链接）——有可用的视频描述预训练权重（需手动下载）。
- PickNet / AdaFrame：无任何权重。
- peek：有 ActivityNet 训练的 `peek_base` 权重（HF，CC BY-NC-SA）。

## 12. 适配 CapERA 的难度：**低–中（约 3 个新文件，不碰框架主体）**

1. `scripts/preprocess_capera.py`：官方 JSON → 仿 `tools/msvd_preprocess.py` 出 `capera_caption_anno_{train,val,test}.pkl` + `vocabulary.txt` + `captions_{val,test}.json`（val 从 train 划小比例用于早停；test 用官方 split）。~150 行。
2. `scripts/extract_capera_feats.py`：mp4（5s@24fps，640×640）→ 逐帧 MaxViT-S（`maxvit_small_tf_224`，timm 0.9.16 已在 GPU 验证）→ `features/capera/maxvit/{video_id}.npz`（opencv-python-headless 已装）。帧率采样（如 1–2 fps → 5–10 帧）由配置定。
3. `xmodaler/datasets/videos/capera.py`：CapERADataset（仿 msvd.py，~100 行；`_sample_frame` 即均匀基线）。
4. 配置文件 `configs/capera/*.yaml`：路径 + MODEL 超参（VOCAB_SIZE、VISUAL_EMBED.IN_DIM=768、MAX_FEAT_NUM 等）。
5. 不改动任何原始数据/视频/标注。

## 13. 适配 WebUAV-3M 的难度：**低–中（与 CapERA 同模式）**

- 同构流程：JPEG 帧 → 逐帧特征 npz → `webuav.py` wrapper（仿 msvd.py/capera.py）→ 配置切换。特征与词表分开管理即可与 CapERA 互通。
- 差异点：① 标注在 `language.txt`（每视频 1 句）而非 JSON——需要一个小解析脚本生成 pkl/JSON；② 无多参考（1 句），CIDEr 参考集语义与 CapERA 不同，跨数据集评测需在报告里说明度量口径；③ 数据量大（Val 28GB / Test 170GB，Train 621GB 等磁盘扩容），特征可先只提取 Val/Test。
- 结论：**dataset 层完全解耦的设计（§Q7 目录结构）天然支持"CapERA 训练 → WebUAV-3M 测试"的跨数据集泛化实验**。

## 14. AdaFrame-style adaptive selection 可插入的位置（xmodaler 内的落点）

- **替换点**：`CapERADataset.__call__` 中 `_sample_frame()`（均匀采样）→ 改为输出**全帧特征 + selector 决策**（或保持 dataset 只出全帧，selector 放模型侧）。
- **模型侧落点**：新增 `xmodaler/modeling/selector/`（policy / global_memory / utility / stop）与一个薄 meta_arch wrapper（`selector_enc_dec.py`，注册进 `META_ARCH_REGISTRY`），内部复用原 `TransformerEncoderDecoder` —— 对框架是"加法"。
- **训练落点**：`xmodaler/engine/` 新增 `SelectorTrainer`（骨架仿 `rl_trainer.py`：采样子集 → 生成 → reward；advantage = r(采样) − r(基线)；REINFORCE + utility 回归 loss）。
- **评测落点**：不变（COCOEvaler），额外记录每视频实际选帧数/计算量统计（新指标日志即可）。

## 15. 我的贡献应插入的位置（Captioning-oriented Adaptive Contribution / Budget Selection）

位于 §14 的 **selector 模块内部**，具体三个组件（下一阶段再设计，本报告只定位）：

1. **Contribution 估计**（替换 AdaFrame 的分类置信度 reward）：逐帧/逐子集的"对 caption 的边际信息增益"——候选方向包括 caption-aware 特征冗余度、与已选集的语义增量、生成分布的变化等（设计阶段再定）。
2. **Budget / utility**（替换 AdaFrame 的 utility 头）：caption 导向的预期收益估计（如预期 CIDEr 增量），支撑 per-video 帧数分配与 lookahead 停止。
3. **Reward 组装**：沿用 PickNet 的 λ_l·语言 + λ_v·视觉多样性 + 预算惩罚结构，把语言项从"整句 CIDEr"升级为"边际贡献"。

目标行为（与用户需求一致）：简单视频少选帧、复杂视频多选帧，预算-质量可用单一旋钮（类 AdaFrame μ）调节。

---

## 附：参考来源

- PickNet 论文页: https://openaccess.thecvf.com/content_ECCV_2018/html/Yangyu_Chen_Less_is_More_ECCV_2018_paper.html
- PickNet arXiv: https://arxiv.org/abs/1803.01457（全文机制引用自 ar5iv: https://ar5iv.labs.arxiv.org/html/1803.01457）
- AdaFrame 论文页: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html
- AdaFrame arXiv: https://arxiv.org/abs/1811.12432（全文机制引用自 ar5iv: https://ar5iv.labs.arxiv.org/html/1811.12432）
- Papers with Code（AdaFrame 无代码）: https://paperswithcode.com/paper/adaframe-adaptive-frame-selection-for-fast
- xmodaler: https://github.com/YehLi/xmodaler
- PEEK: https://github.com/momentslab/peek（论文 arXiv:2605.31029）
- video-captioning-models-in-Pytorch: https://github.com/nasib-ullah/video-captioning-models-in-Pytorch
- MART: https://github.com/jayleicn/recurrent-transformer
- self-critical.pytorch: https://github.com/ruotianluo/self-critical.pytorch
- video-caption.pytorch: https://github.com/xiadingZ/video-caption.pytorch
- FrameGlimpse: https://github.com/syyeung/frameglimpses
- PickNet 中文解读（仅论文解读，无代码）: https://zhuanlan.zhihu.com/p/124632860
