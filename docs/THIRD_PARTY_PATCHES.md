# third_party/xmodaler 补丁记录

规则（来自用户指示 §九）：尽量不改第三方代码；必须改则最小局部、不改变原 MSVD pipeline 行为、记录原因。

## 补丁 1（2026-09-29）

- 文件：`third_party/xmodaler/xmodaler/tokenization/tokenization_bert.py:25`
- 改动：`from pytorch_transformers.tokenization_utils import PreTrainedTokenizer`
      → `from transformers.tokenization_utils import PreTrainedTokenizer`
- 原因：`xmodaler/datasets/__init__.py` 导入链（mscoco_bert → tokenization_bert）在 import 时解析该旧包名；`pytorch_transformers` 在 2019 年已更名为 `transformers`，本环境装的是 transformers 4.36.2。若不打补丁，**import xmodaler.datasets 直接失败**，任何配置都无法加载。
- 影响面：仅 import 期解析；该文件内 vendored 的 `BertTokenizer` 类只在 `INFERENCE.VOCAB == 'BERT'` 时才实例化，本 baseline 使用词表文件路径（`INFERENCE.VOCAB = vocabulary.txt`），**不会走到该类的运行路径**，对 MSVD/COCO BERT 用法行为零改变（其运行时兼容性问题属上游代码与 transformers 4.x 的历史问题，不在本实验范围）。
- 验证：transformers 4.36.2 确认存在 `transformers.tokenization_utils.PreTrainedTokenizer`；冒烟测试全链路 import 通过。

## 补丁 2（2026-09-29，环境级，非 xmodaler 文件）

- 位置：`scripts/train_capera.py`（项目侧入口，非 third_party）
- 改动：替换 `pycocoevalcap.eval.Spice` 为 no-op（`_DummySpice`），从评测中禁用 SPICE。
- 原因：pycocoevalcap 的 `COCOEvalCap.evaluate()` 无条件构造 `Spice()`，其构造时会下载 stanford-corenlp-3.6.0（384 MB，本机直连极慢）且评分需要 JVM。本实验只报告 BLEU-4/METEOR/ROUGE-L/CIDEr，不需要 SPICE。
- 影响面：仅本项目评测入口；COCOEvalCap 输出中 SPICE 恒为 0.0（可忽略）。

## 补丁 3（2026-09-29，项目侧，非 xmodaler 文件）

- 位置：`scripts/train_capera.py` 中 `CapEraTrainer`（注册进 ENGINE_REGISTRY，`ENGINE.NAME: CapEraTrainer`）
- 改动：继承 `DefaultTrainer` 并覆写 `test()`，唯一差异：空生成 caption 用 `'.'` 占位。
- 原因：训练早期模型会对部分视频首个 token 就输出 EOS，`decode_sequence` 得到空串，PTB 分词后为 0 token，触发 pycocoevalcap BLEU 的 `assert len(hypo) == 1` 崩溃。占位保证每个视频恰好 1 条预测。

## 补丁 4（2026-09-29，数据协议，非 xmodaler 文件）

- 位置：`scripts/preprocess_capera.py`
- 改动：val/test 的 anno pkl 每视频**只保留 1 条** entry（train 保持 5 条/视频）；COCO 参考 JSON 保持 5 句/视频不变。
- 原因：val/test 是生成模式不需要 caption；若每视频 5 条 entry，评测时每个视频会生成 5 条预测，同样触发 `len(hypo)==1` 断言（实测崩溃证据在 logs/train_capera_xe.log v4 运行）。

## 环境依赖安装记录（2026-09-29，uavcap env）

- 为运行 xmodaler 补装：`tabulate`、`termcolor`、`fvcore`、`omegaconf`、`portalocker`、`psutil`、`json-lines`、`jsonlines`（xmodaler 上游 `datasets/__init__.py` 导入链需要）、`setuptools<81`（torch 2.1.2 的 `cpp_extension` 依赖 `pkg_resources`，setuptools ≥81 已移除该模块）。
- 未升级/未改动：numpy 1.26.4、torch 2.1.2+cu121、torchvision 0.16.2、transformers 4.36.2、timm 0.9.16。
