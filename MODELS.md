# CRUX 模型版本表

命名规则：`crux-hold-seg-v<major>.<minor>.<patch>-<导出分辨率>-<精度>.onnx`

- major：训练数据/纠正阶段变化（v0 = Kaggle research baseline；v1 = 含 realpic correction 的 research line）
- minor：训练配置变化（分辨率/增强/模型架构）
- patch：同一配置的修复性重训

`Runtime default` 只表示当前研发工具默认使用的模型；`Distribution` 表示是否允许随 App 分发。两者不能等同。

| 版本 | 训练数据 | 分辨率 | 增强 | val mAP_50_95 | ONNX 文件 | Runtime default | Distribution | 状态 |
|---|---|---|---|---|---|---|---|---|
| v0.1.0 | Kaggle bh (12 train) | 384 | 无 | ~0.29 | ~~`output/onnx/...`~~ | no | no | baseline（**已清理**） |
| v0.2.0 | Kaggle bh (12 train) | 640 | 无 | ~0.29 | ~~`output/onnx_640/...`~~ | no | no | 细掩码验证（**已清理**） |
| **v0.3.0** | Kaggle bh (12 train) | 640 | AUG_CONSERVATIVE, 130ep | **0.276** | `output/onnx_640_aug/crux-hold-seg-v0.3.0-648-fp16.onnx` | no | no | 上代 research baseline |
| v0.3.0-fp32 | 同上 | 640 | 同上 | 同上 | `output/onnx_640_aug_fp32/crux-hold-seg-v0.3.0-648-fp32.onnx` | no | no | INT8 量化源 |
| **v1.0.1** | Kaggle 12 + realpic 7 (×4 重采样) | 640 | 纠正微调: lr 1e-5, 10ep | **0.236** (Kaggle val) | `output/onnx_v101/crux-hold-seg-v1.0.1-648-fp16.onnx` | **yes (research)** | **no** | correction research checkpoint；非 production |

### realpic test2017 / correction-set evaluation（不是独立泛化测试）

7 张 realpic 共 333 个人工标注，现位于 `data/crux-dataset/test2017`。它们曾参与
v1.0.1 correction fine-tuning；以下指标只用于历史纠错对比，不能命名为独立
validation/test，也不能证明 production 泛化能力。

| 版本 | P | R | F1 | FP | FN |
|---|---|---|---|---|---|
| v0.3.0 | 0.71 | 1.00 | 0.83 | 137 | 1 |
| v1.0.0 (混合) | 0.74 | 0.91 | 0.82 | 108 | 29 |
| **v1.0.1 (纠正)** | **0.83** | **0.97** | **0.89** | **68** | **10** |

v1.0.1 = `crux-dataset-v101` 中 realpic 重采样 ×4 + lr 1e-5 + 10 epochs
从 v0.3.0 微调。**纠正式增量 > 混合训练**（用户标注是纠错信号，不是平等样本）。

## 数据集语义

- `data/crux-dataset/train2017`：12 张 Kaggle research 训练图。
- `data/crux-dataset/test2017`：7 张 realpic GT 图；保留为评估集目录，但因历史
  上参与 v1.0.1 训练，只能叫 correction/evaluation set。
- `data/crux-dataset-v101`：v1.0.1 历史 correction-train 复现集，不能作为独立测试集。

## 训练 checkpoint 对应

| 版本 | ckpt 目录 | research pth |
|---|---|---|
| v0.1.0/v0.2.0 | ~~`output/rfdetr_seg_small/`~~（已清理） | — |
| v0.3.0 | `output/rfdetr_seg_small_aug/` | `crux-hold-seg-research.pth` |
| v1.0.1 | `output/rfdetr_seg_small_v101/` | `crux-hold-seg-research.pth` |

## 已知限制

- 数据仍不足以代表独立 production 泛化；v1.0.1 的 7 张 realpic 同时参与 correction fine-tuning。
- production-distributable 模型必须使用自有或明确授权数据，并建立独立 `realpic_test`。
- v1.0.1 的 static INT8 仅为部署候选；FP16 是 accuracy reference，需经过端侧 parity、recall 和 correction-count Gate。

## 推理脚本默认指向

需要模型推理的工具默认使用 research runtime reference：

`output/onnx_v101/crux-hold-seg-v1.0.1-648-fp16.onnx`
