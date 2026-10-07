# PickNet Code Provenance Audit

日期：2026-09-30 ｜ 范围：本项目全部代码 + `/root/autodl-tmp/repo_audit/` + 公开来源复核 ｜ 状态：只读审计，未修改/删除任何文件，未训练

---

## 1. Official repository status

**No official PickNet implementation was found.**

检索证据（2026-09-30 复核，此前 2026-09-29 首次检索结论一致）：

| 来源 | 结果 |
|---|---|
| ECCV 2018 官方论文页（openaccess.thecvf.com，HTTP 200 复核） | 仅 `[pdf]` / `[arXiv]` / `[bibtex]` 链接，**无任何 code/project/homepage 链接** |
| arXiv abs 页（1803.01457，复核） | 无代码链接 |
| arXiv 全文（ar5iv HTML） | 全文无 github 链接（仅引用工具库 jcjohnson/cnn-benchmarks、tylin/coco-caption） |
| HuggingFace Papers API（`/api/papers/1803.01457`，复核） | `implementations: 0` |
| Papers with Code | 无实现注册（该站现重定向至 HF Papers，以上为准） |
| Semantic Scholar API | 请求被限流（429，已尝试） |
| OpenBayes Trends 论文页 | 仅平台方链接 github.com/hyperai，无实现 |
| GitHub 仓库搜索 API（复核，5 组查询） | `picknet`：12 个仓库**全部为地震学震相拾取/CSV 工具**；`picknet+video+captioning`：0；`picking+informative+frames`：0；`1803.01457`：0；`less+is+more+video+captioning`：3 个无关仓库 |
| Gitee 搜索 API（复核） | `[]` |
| grep.app / Sourcegraph 代码级搜索 | 被 WAF 拦截（已记录尝试） |
| WebSearch（2026-09-30，多组查询含 2024/2025 复现检索） | 仅论文链接，无任何实现仓库 |

## 2. Author repository search

| 作者 | 检索 | 结果 |
|---|---|---|
| Yangyu Chen（一作） | GitHub 用户搜索 `yangyu`、`yangyu+chen` | 无任何账号与 ECCV 2018 PickNet 相关（列出的 cyyself/claytoncyy 等均无关） |
| Shuhui Wang（IIE CAS） | GitHub 用户搜索 `shuhui+wang` → 2 个候选账号，逐一检查全部仓库 | `wsh110714`（web/游戏开发）、`shuhuiwang1`（生物 ML 练习）——**均非本论文作者，无 PickNet 仓库** |
| Weigang Zhang | GitHub 用户搜索 `weigang+zhang+cas` | 0 结果 |
| Qingming Huang | GitHub 用户搜索 `qingming+huang` | 1 个无关账号 |
| 作者主页/机构页 | WebSearch | 未发现任何代码发布页 |

## 3. Third-party implementation search

**无可信第三方实现。** 逐条排除：

- 全部名为 "PickNet" 的 GitHub 仓库（12 个）经描述逐一核对：MrXiaoXiao/PickNet_keras 与 Dengda98/PickNet_* 是**地震波初至拾取**（STEAD/INSTANCE 数据集），其余为 CSV 工具、Netflix 玩具项目等——**与本论文无关**；
- 无任何仓库实现"顺序 Pick/Drop + video captioning"的 ECCV 2018 结构；
- 无任何仓库引用 arXiv:1803.01457 作为实现来源（API 查询 total=0）；
- 中文社区（知乎/CSDN/Gitee）仅有论文解读文章，无代码。

## 4. Current project provenance（本项目内代码的真实来源）

**项目中不存在任何外来 PickNet 代码。** 事实清单：

1. **无 PickNet git 仓库**：项目根目录不是 git 仓库（`git status` → not a git repository）；全项目唯一 `.git` 是 `third_party/xmodaler/.git`（remote = `https://github.com/YehLi/xmodaler.git`，commit `ed1f5590`，2023-02-28，上游 xmodaler 框架，与 PickNet 无关）；
2. **`/root/autodl-tmp/repo_audit/` 三个克隆均非 PickNet**：`recurrent-transformer`（jayleicn/MART，2020-12-04）、`video-captioning-models-in-Pytorch`（nasib-ullah，2023-07-30）、`xmodaler`（YehLi，2023-02-28）；
3. **selector 相关源码均为本次会话新写**（文件系统时间戳）：
   - `models/selectors/base_selector.py` — 创建于 **2026-09-30 09:12:44**
   - `models/selectors/picknet_style_selector.py` — 创建于 **2026-09-30 09:12:55**
   - `models/selector_enc_dec.py` — 创建于 **2026-09-30 09:13:10**
   - 作者：Claude Code（本会话），按用户阶段二规格 + 论文 §3.2 机制描述编写；
4. 源码 docstring 自述（`picknet_style_selector.py:1-9`）："NOT an official PickNet reproduction: we use the pre-extracted 768-d frame features ... rather than PickNet's original low-resolution grayscale glance input, and v1 trains through the captioning XE loss with a straight-through estimator rather than PickNet's REINFORCE protocol"；
5. `docs/PickNet_Style_Implementation.md` §开头亦有精确声明（非官方复现、adapted to our pipeline）。

> **结论：当前 PickNet-style selector is a paper-inspired reimplementation, not an official PickNet codebase.**

## 5. Exact difference: official PickNet vs our paper-inspired implementation

| 维度 | Official PickNet（论文描述） | 我们的 PickNet-style 实现 |
|---|---|---|
| 代码来源 | **无公开代码** | 本次会话新写（按论文 §3 机制描述） |
| 选择器输入 | 56×56 灰度 glance 差分图（展平） | 预提取 MaxViT-S 768-d 帧特征（与 captioner 输入同源） |
| 决策网络 | 2 层前馈网络（无循环状态） | GRUCell(768→256) + MLP → 2 logits（带顺序状态） |
| 决策方式 | 顺序扫描，逐帧伯努利 Pick/Drop | 同：顺序扫描、逐帧 Pick/Drop、首帧强制 PICK、变长选择 |
| 训练 | 三阶段：监督 XE → REINFORCE（CIDEr+diversity reward，self-critical 基线）→ 近似联合 | 单阶段 XE + STE + 可微期望预算正则（**非 RL**） |
| 帧数约束 | N_min=3 / N_max（训练中收缩）硬区间 + 罚 R⁻ | 软正则（期望选择率目标 0.5） |
| Captioner | LSTM 编码 + GRU 解码 | xmodaler Transformer Encoder/Decoder（与 uniform baseline 同骨架） |
| 数据集 | MSVD / MSR-VTT | CapERA |

## 6. What can legitimately be claimed（论文中可合法声明）

1. "We implement a **PickNet-style** sequential Pick/Drop frame selection baseline **based on the PickNet paper** (Chen et al., ECCV 2018)";
2. "The baseline follows the paper's core selection protocol: sequential frame scanning, per-frame Pick/Drop decisions, first frame forced selected, video-dependent number of selected frames";
3. "It is adapted to our feature-based captioning pipeline (768-d frame features, Transformer captioner), and trained with a captioning-aware objective rather than the original REINFORCE protocol";
4. 任何关于我们自己 selector 的实验结果（帧数统计、指标、开销）都可报告——它们是本实现的真实测量。

## 7. What must NOT be claimed（论文中禁止声明）

1. ❌ 不得声称"我们复现了官方 PickNet"或"使用官方 PickNet 代码"；
2. ❌ 不得声称"our PickNet baseline reproduces the original paper's results"（无官方代码可比对，且训练协议不同）；
3. ❌ 不得将本实现与 PickNet 论文的 MSVD/MSR-VTT 数字直接对比并暗示"相同方法"（输入特征、captioner、训练协议、数据集均不同）；
4. ❌ 不得声称实现了原论文的 RL 训练协议（REINFORCE / CIDEr reward / diversity reward）——当前版本没有；
5. ❌ 不得把 seismic PickNet 等无关同名项目列为本方法的前作/实现来源。

## 8. 最终结论

**B. 没有找到可信官方代码，因此当前实现只能称为：**

> **"PickNet-style baseline implemented based on the PickNet paper"**

（即 `models/selectors/picknet_style_selector.py` 是 paper-inspired reimplementation；项目内不存在任何 PickNet 官方/第三方代码；后续论文写作必须使用 §6 的措辞并遵守 §7 的禁止项。）

---

附：本审计与 `docs/PickNet_Code_Audit_Report.md`（2026-09-29 框架选型审计）的关系——后者确认了"无公开代码→选用 xmodaler 作框架"，本报告确认了"当前项目内 selector 代码的 provenance = 本次会话新写、paper-inspired"。两份报告结论一致。
